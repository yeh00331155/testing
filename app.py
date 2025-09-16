import os
import io
import json
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ---------- Load pipeline.pkl ----------
MODEL_PATH = os.environ.get("PIPELINE_PATH", "pipeline.pkl")

if not os.path.exists(MODEL_PATH):
    PIPELINE = None
else:
    with open(MODEL_PATH, "rb") as f:
        PIPELINE = pickle.load(f)

def _require_pipeline():
    if PIPELINE is None:
        raise HTTPException(status_code=500, detail="pipeline.pkl 未找到，請將檔案放在同目錄或設定環境變數 PIPELINE_PATH")

def get_obj(name: str):
    _require_pipeline()
    if name not in PIPELINE:
        raise HTTPException(status_code=500, detail=f"pipeline.pkl 缺少必要物件：{name}")
    return PIPELINE[name]

def try_get_obj(name: str, default=None):
    return None if (PIPELINE is None or name not in PIPELINE) else PIPELINE[name]

# 預設常數（可放在 pipeline.pkl['defaults']）
DEFAULTS = try_get_obj("defaults", {}) or {}

# Optional: 歷史事故密度對照表（同資料夾放 location_counts.csv，含欄位：Location Name,count）
LOC_COUNT_MAP: Optional[Dict[str, int]] = None
if os.path.exists("location_counts.csv"):
    _df = pd.read_csv("location_counts.csv")
    if "Location Name" in _df.columns and "count" in _df.columns:
        LOC_COUNT_MAP = dict(zip(_df["Location Name"], _df["count"]))

# ---------- Pydantic schemas ----------
class PredictItem(BaseModel):
    Date: str = Field(..., description="YYYYMMDD")
    Time: str = Field(..., description="HHMMSS（不足左補零）")
    Cause_Analysis: Optional[str] = Field(None, alias="Cause Analysis")
    Road_Condition: Optional[str] = Field(None, alias="Road Condition")
    loc_acc_count: Optional[Union[int, float]] = 0
    Location_Name: Optional[str] = Field(None, alias="Location Name")

    class Config:
        populate_by_name = True
        extra = "allow"  # 允許其餘 cat_feats 直接帶上來

    @validator("Date")
    def _valid_date(cls, v):
        if not (len(v) == 8 and v.isdigit()):
            raise ValueError("Date 需為 YYYYMMDD")
        return v

    @validator("Time")
    def _valid_time(cls, v):
        if not (len(v) == 6 and v.isdigit()):
            raise ValueError("Time 需為 HHMMSS")
        return v

class PredictRequest(BaseModel):
    data: Union[PredictItem, List[PredictItem]]

# ---------- Feature builders ----------
def parse_datetime(date_str: str, time_str: str):
    yyyy, mm, dd = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    HH, MM, SS = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
    import datetime as _dt
    dt = _dt.datetime(yyyy, mm, dd, HH, MM, SS)
    return {"Hour": HH, "Month": mm, "Weekday": dt.weekday()}  # Monday=0

def is_weekend_func(weekday: int) -> int:
    return 1 if weekday >= 5 else 0  # Sat(5), Sun(6)

def is_peak_func(hour: int) -> int:
    return 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0

def loc_density(loc_name: Optional[str], given: Optional[Union[int, float]]):
    if given is not None:
        try:
            return float(given)
        except Exception:
            pass
    if LOC_COUNT_MAP is not None and loc_name is not None:
        return float(LOC_COUNT_MAP.get(loc_name, 0))
    return 0.0

def build_feature_frame(items: List[PredictItem]) -> pd.DataFrame:
    _require_pipeline()

    tfv = get_obj("tfv")
    le  = try_get_obj("le")
    scaler = try_get_obj("scaler")
    num_feats = get_obj("num_feats")
    tfidf_cols = get_obj("tfidf_cols")
    cat_feats = get_obj("cat_feats")
    ohe = try_get_obj("ohe")

    rows, texts = [], []

    for it in items:
        base = parse_datetime(it.Date, it.Time)
        base["is_weekend"] = is_weekend_func(base["Weekday"])
        base["is_peak"] = is_peak_func(base["Hour"])
        base["loc_acc_count"] = loc_density(it.Location_Name, it.loc_acc_count)

        # RoadCond_Ord：優先用輸入，其次 DEFAULTS["Road Condition"]，最後 fallback 至 le.classes_[0]
        rc = (it.Road_Condition if it.Road_Condition is not None else DEFAULTS.get("Road Condition", None))
        rc = (rc or "").strip()
        if le is not None:
            try:
                base["RoadCond_Ord"] = int(le.transform([rc])[0])
            except Exception:
                classes = list(map(str, getattr(le, "classes_", [])))
                if classes:
                    base["RoadCond_Ord"] = int(le.transform([classes[0]])[0])  # fallback to first known class
                else:
                    raise HTTPException(status_code=422, detail="Road Condition 類別資訊缺失，請檢查 pipeline.pkl 的 le.classes_")
        else:
            base["RoadCond_Ord"] = 0  # 沒 encoder 就先 0（建議仍在 pipeline.pkl 放 le）

        # 其餘 cat_feats：缺則用 DEFAULTS，最後給空字串
        cat_row = {}
        for c in cat_feats:
            if c in ["Road Condition", "Road_Condition"]:
                continue
            val = getattr(it, c, None)
            if val is None and hasattr(it, "__dict__"):
                val = it.__dict__.get(c, None)
            if val is None:
                val = DEFAULTS.get(c, None)
            cat_row[c] = "" if val is None else str(val)
        base.update(cat_row)

        texts.append((it.Cause_Analysis or "").strip())
        rows.append(base)

    df = pd.DataFrame(rows)

    # 確保 num_feats 存在
    for nf in num_feats:
        if nf not in df.columns:
            df[nf] = 0

    # TF-IDF
    tf_mat = tfv.transform([t for t in texts])
    try:
        tf_df = pd.DataFrame(tf_mat.toarray(), columns=tfidf_cols)
    except Exception:
        tf_df = pd.DataFrame(tf_mat.toarray(), columns=[f"tfidf_{i}" for i in range(tf_mat.shape[1])])

    # OneHotEncoder（若有）
    if ohe is not None:
        cats = [c for c in cat_feats if c not in ["Road Condition", "Road_Condition"]]
        if cats:
            ohe_mat = ohe.transform(df[cats])
            if hasattr(ohe, "get_feature_names_out"):
                ohe_cols = list(ohe.get_feature_names_out(cats))
            else:
                ohe_cols = []
                for i, c in enumerate(cats):
                    for v in ohe.categories_[i]:
                        ohe_cols.append(f"{c}__{v}")
            ohe_df = pd.DataFrame(ohe_mat.toarray() if hasattr(ohe_mat, "toarray") else ohe_mat, columns=ohe_cols)
            df = pd.concat([df.drop(columns=cats, errors="ignore"), ohe_df], axis=1)

    # 數值部分 + scaler
    X_num = df[num_feats].copy()
    if scaler is not None:
        X_num = pd.DataFrame(scaler.transform(X_num), columns=num_feats)

    X = pd.concat([X_num.reset_index(drop=True), tf_df.reset_index(drop=True)], axis=1)
    return X

def predict_proba_with_model(X: pd.DataFrame) -> Dict[str, Any]:
    model = get_obj("model")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = [str(i) for i in range(proba.shape[1])]
        probs = [dict(zip(map(str, classes), map(float, row))) for row in proba]
    else:
        probs = [None] * len(X)
    pred = model.predict(X)
    return {"pred": [p if isinstance(p, (str, int)) else str(p) for p in pred], "probs": probs}

# ---------- FastAPI app ----------
app = FastAPI(
    title="Traffic Accident Prediction API (MVP)",
    description="以 pipeline.pkl 驅動；支援單筆 JSON 與多筆 Excel。側欄僅地點/日期/時間可由前端實作。",
    version="0.2.0"
)

@app.get("/health")
def health():
    return {"ok": PIPELINE is not None}

@app.get("/schema")
def schema():
    _require_pipeline()
    cat_feats = get_obj("cat_feats")
    le = try_get_obj("le")
    le_classes = list(map(str, getattr(le, "classes_", []))) if le is not None else []
    return {
        "required_fields": ["Date", "Time", "Cause Analysis", "Road Condition"] + [c for c in cat_feats if c not in ["Road Condition", "Road_Condition"]],
        "optional_fields": ["loc_acc_count", "Location Name"],
        "derived_fields": ["Hour", "Month", "Weekday", "is_weekend", "is_peak", "RoadCond_Ord"] + list(get_obj("tfidf_cols")),
        "categorical_features": cat_feats,
        "road_condition_allowed": le_classes,
        "defaults": DEFAULTS,
        "notes": "其他類別欄位請依 cat_feats 欄名提供字串值；/schema 的 allowed 供前端下拉使用。"
    }

@app.post("/predict")
def predict(req: PredictRequest):
    _require_pipeline()
    items = req.data if isinstance(req.data, list) else [req.data]
    X = build_feature_frame(items)
    out = predict_proba_with_model(X)
    return JSONResponse({
        "prediction": out["pred"][0] if len(out["pred"]) == 1 else out["pred"],
        "probabilities": out["probs"][0] if len(out["probs"]) == 1 else out["probs"],
    })

@app.post("/batch")
async def batch(file: UploadFile = File(..., description="上傳 .xlsx，工作表含 MVP 所需欄位")):
    _require_pipeline()
    if not (file.filename.lower().endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="請上傳 .xlsx 檔")
    content = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"讀取 Excel 失敗：{e}")

    records = df.to_dict(orient="records")
    items = [PredictItem(**r) for r in records]
    X = build_feature_frame(items)
    out = predict_proba_with_model(X)
    res = pd.DataFrame({"prediction": out["pred"]})
    if out["probs"][0] is not None:
        probs_df = pd.DataFrame(out["probs"])
        res = pd.concat([res, probs_df], axis=1)

    return JSONResponse({
        "n_rows": len(res),
        "rows": json.loads(res.to_json(orient="records", force_ascii=False)),
    })
