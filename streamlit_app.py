
import os
import json
import requests
import pandas as pd
import streamlit as st
from datetime import date, time

st.set_page_config(page_title="事故預測（MVP）", page_icon="🚦", layout="wide")

st.sidebar.header("參數（最小版）")
loc = st.sidebar.text_input("地點（Location Name）", "")
d = st.sidebar.date_input("日期（Date）", value=date.today(), format="YYYY-MM-DD")
t = st.sidebar.time_input("時間（Time）", value=time(8,30,0), step=60)

api_base = "http://localhost:8000"

st.title("🚦 交通事故預測（MVP）")
st.caption("後端自動計算週末/尖峰/歷史密度與 TF‑IDF，並用 pipeline.pkl 預測。只需地點、日期、時間。")

with st.expander("查看後端 /schema（可用類別與預設）"):
    try:
        sch = requests.get(f"{api_base}/schema", timeout=5).json()
        st.json(sch, expanded=False)
    except Exception as e:
        st.warning(f"無法連線到 API /schema：{e}")

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("單筆預測")
    if st.button("送出預測", type="primary"):
        if not loc:
            st.error("請輸入地點（Location Name）。")
        else:
            ds = d.strftime("%Y%m%d")
            ts = t.strftime("%H%M%S")
            payload = {
                "data": {
                    "Date": ds,
                    "Time": ts,
                    "Cause Analysis": "",
                    "Road Condition": None,
                    "Location Name": loc
                }
            }
            try:
                r = requests.post(f"{api_base}/predict", json=payload, timeout=10)
                if r.status_code == 200:
                    st.success("成功！")
                    st.json(r.json())
                else:
                    st.error(f"API 回應 {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"請求失敗：{e}")

with col2:
    st.subheader("批次上傳 Excel")
    up = st.file_uploader("上傳 .xlsx", type=["xlsx"])
    if up is not None and st.button("批次預測", key="batch"):
        try:
            files = {"file": (up.name, up.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            r = requests.post(f"{api_base}/batch", files=files, timeout=30)
            if r.status_code == 200:
                data = r.json().get("rows", [])
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("沒有資料列。")
            else:
                st.error(f"API 回應 {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"請求失敗：{e}")

st.divider()
st.caption(f"API_BASE = {api_base}（可用 Streamlit secrets 或環境變數覆寫）")
