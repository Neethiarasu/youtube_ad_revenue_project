# app.py 
import streamlit as st
import pandas as pd
import joblib
from datetime import date

st.title("YouTube Monetization Predictor")

# --- simple inputs ---
video_id = st.text_input("video_id", value="vid_minimal_001")
views = st.number_input("views", value=20000, min_value=0)
likes = st.number_input("likes", value=1000, min_value=0)
comments = st.number_input("comments", value=100, min_value=0)
watch_time = st.number_input("watch_time_minutes", value=50000, min_value=0)
video_length = st.number_input("video_length_minutes", value=10, min_value=1)
subs = st.number_input("subscribers", value=50000, min_value=0)
category = st.selectbox("category", ["Education","Music","Gaming","Tech","Vlog","Sports"])
device = st.selectbox("device", ["Mobile","Desktop","Tablet","TV"])
country = st.selectbox("country", ["US","IN","UK","CA","AU","DE"])
report_date = st.date_input("date", value=date.today())

if st.button("Predict (fresh)"):
    # load pipeline here (fresh each click)
    pipe = joblib.load("final_ridge_pipeline.joblib")

    # get exact names the pipeline expects
    pre = pipe.named_steps["preprocessor"]
    expected = list(pre.feature_names_in_)
    st.write("EXPECTED from pipeline:", expected)

    # build input dict exactly matching expected
    row = {}
    for c in expected:
        if c == "video_id":
            row[c] = str(video_id) or "vid_minimal_001"
        elif c == "date":
            row[c] = pd.to_datetime(report_date).strftime("%Y-%m-%d")
        elif c == "views":
            row[c] = int(views)
        elif c == "likes":
            row[c] = int(likes)
        elif c == "comments":
            row[c] = int(comments)
        elif c == "watch_time_minutes":
            row[c] = float(watch_time)
        elif c == "video_length_minutes":
            row[c] = float(video_length)
        elif c == "subscribers":
            row[c] = int(subs)
        elif c in ("category","device","country"):
            row[c] = str(locals().get(c))
        elif c == "engagement_rate":
            row[c] = (likes + comments) / views if views>0 else 0.0
        elif c == "likes_per_view":
            row[c] = likes / views if views>0 else 0.0
        elif c == "comments_per_view":
            row[c] = comments / views if views>0 else 0.0
        elif c == "watch_time_per_view":
            row[c] = watch_time / views if views>0 else 0.0
        elif c == "view_length_ratio":
            row[c] = (watch_time / video_length) if video_length>0 else 0.0
        elif c == "video_age_days":
            row[c] = float((pd.to_datetime("today") - pd.to_datetime(report_date)).days)
        elif c == "views_per_day":
            days = (pd.to_datetime("today") - pd.to_datetime(report_date)).days
            days = days if days>0 else 1
            row[c] = float(views)/days
        elif c == "is_long_video":
            row[c] = int(video_length >= 20)
        else:
            row[c] = 0

    df = pd.DataFrame([row], columns=expected)
    st.write("INPUT columns in this run:", df.columns.tolist())
    st.dataframe(df)

    # predict
    try:
        pred = float(pipe.predict(df)[0])
        st.success(f"Estimated ad revenue: ${pred:.2f} USD")
    except Exception as e:
        st.error("Prediction error:")
        st.text(str(e))
