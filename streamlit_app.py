import pickle
import pandas as pd
import streamlit as st
from datetime import date, time, datetime

# ========= 載入 pipeline.pkl =========
@st.cache_resource
def load_pipeline():
    with open("pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)
    return pipe

PIPELINE = load_pipeline()

tfv = PIPELINE["tfv"]
model = PIPELINE["model"]
scaler = PIPELINE.get("scaler", None)
num_feats = PIPELINE["num_feats"]
tfidf_cols = PIPELINE["tfidf_cols"]
cat_feats = PIPELINE["cat_feats"]
defaults = PIPELINE.get("defaults", {})

# ========= 自動特徵工程 =========
def build_features(date_str, time_str, location):
    # 轉日期時間
    yyyy, mm, dd = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    HH, MM, SS = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
    dt = datetime(yyyy, mm, dd, HH, MM, SS)

    feat = {
        "Hour": HH,
        "Month": mm,
        "Weekday": dt.weekday(),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "is_peak": 1 if (7 <= HH <= 9) or (17 <= HH <= 19) else 0,
        "loc_acc_count": 0,  # 沒有 loc_acc_count.csv 時，預設 0
    }

    # 其他分類特徵 → 用 defaults 填入
    for c in cat_feats:
        if c in ["Road Condition", "Road_Condition"]:
            feat["RoadCond_Ord"] = defaults.get("RoadCond_Ord", 0)
        else:
            feat[c] = defaults.get(c, "NA")

    # Cause Analysis → 這裡先當空字串
    cause_text = ""

    # TF-IDF
    tf_mat = tfv.transform([cause_text])
    tf_df = pd.DataFrame(tf_mat.toarray(), columns=tfidf_cols)

    # 數值欄位
    X_num = pd.DataFrame([feat], columns=num_feats)
    if scaler is not None:
        X_num = pd.DataFrame(scaler.transform(X_num), columns=num_feats)

    X = pd.concat([X_num.reset_index(drop=True), tf_df.reset_index(drop=True)], axis=1)
    return X

# ========= Streamlit UI =========
st.set_page_config(page_title="交通事故預測", page_icon="🚦", layout="wide")

st.sidebar.header("輸入參數（簡化版）")
loc = st.sidebar.text_input("地點 (Location Name)", "斗六市-鎮南路/民生路")
d = st.sidebar.date_input("日期 (Date)", value=date.today())
t = st.sidebar.time_input("時間 (Time)", value=time(8, 30))

st.title("🚦 交通事故預測（MVP）")
st.caption("只需地點 / 日期 / 時間，其餘特徵由 pipeline.pkl 的 defaults 自動補齊。")

if st.button("送出預測", type="primary"):
    ds = d.strftime("%Y%m%d")
    ts = t.strftime("%H%M%S")
    X = build_features(ds, ts, loc)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = getattr(model, "classes_", [str(i) for i in range(probs.shape[1])])
        pred = model.predict(X)[0]
        st.success(f"模型預測結果：{pred}")
        st.subheader("各類別機率")
        st.dataframe(pd.DataFrame(probs, columns=classes))
    else:
        pred = model.predict(X)[0]
        st.success(f"模型預測結果：{pred}（此模型無機率輸出）")
