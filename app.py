
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Online Shopper Purchase Predictor", layout="wide", page_icon="🛒")

@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/best_model.pkl")
    preprocessor = joblib.load("outputs/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

st.title("🛒 Online Shopper Purchase Intention Predictor")
st.markdown("Predict whether a browsing session will end in a purchase, using a trained ML model.")

st.sidebar.header("Session Input Features")

def user_input():
    admin = st.sidebar.number_input("Administrative pages viewed", 0, 30, 0)
    admin_dur = st.sidebar.number_input("Administrative duration (s)", 0.0, 3000.0, 0.0)
    info = st.sidebar.number_input("Informational pages viewed", 0, 30, 0)
    info_dur = st.sidebar.number_input("Informational duration (s)", 0.0, 3000.0, 0.0)
    product = st.sidebar.number_input("Product related pages viewed", 0, 700, 1)
    product_dur = st.sidebar.number_input("Product related duration (s)", 0.0, 15000.0, 0.0)
    bounce = st.sidebar.slider("Bounce rate", 0.0, 0.2, 0.02)
    exit_rate = st.sidebar.slider("Exit rate", 0.0, 0.2, 0.02)
    page_values = st.sidebar.number_input("Page values", 0.0, 400.0, 0.0)
    special_day = st.sidebar.slider("Special day closeness", 0.0, 1.0, 0.0)
    weekend = st.sidebar.checkbox("Weekend session")
    visitor_type = st.sidebar.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
    month = st.sidebar.selectbox("Month", ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])
    os_ = st.sidebar.number_input("Operating system code", 1, 8, 1)
    browser = st.sidebar.number_input("Browser code", 1, 13, 1)
    region = st.sidebar.number_input("Region code", 1, 9, 1)
    traffic = st.sidebar.number_input("Traffic type code", 1, 20, 1)

    total_pages = admin + info + product
    total_duration = admin_dur + info_dur + product_dur
    avg_dur_page = total_duration / total_pages if total_pages > 0 else 0
    product_ratio = product / total_pages if total_pages > 0 else 0
    page_eff = page_values / (exit_rate + 1e-5)
    bounce_exit_ratio = bounce / (exit_rate + 1e-5)
    weekend_returning = int(weekend and visitor_type == "Returning_Visitor")
    is_holiday = int(month in ["Nov", "Dec", "May"])

    data = {
        "Administrative": admin, "Administrative_Duration": admin_dur,
        "Informational": info, "Informational_Duration": info_dur,
        "ProductRelated": product, "ProductRelated_Duration": product_dur,
        "BounceRates": bounce, "ExitRates": exit_rate, "PageValues": page_values,
        "SpecialDay": special_day, "OperatingSystems": os_, "Browser": browser,
        "Region": region, "TrafficType": traffic, "Weekend": int(weekend),
        "Total_Pages_Viewed": total_pages, "Total_Duration": total_duration,
        "Avg_Duration_Per_Page": avg_dur_page, "ProductRelated_Ratio": product_ratio,
        "Page_Efficiency": page_eff, "Bounce_Exit_Ratio": bounce_exit_ratio,
        "Weekend_Returning": weekend_returning, "Is_Holiday_Season": is_holiday,
    }
    for m in ["Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]:
        data[f"Month_{m}"] = int(month == m)
    for v in ["New_Visitor", "Other"]:
        data[f"VisitorType_{v}"] = int(visitor_type == v)

    return pd.DataFrame([data])

input_df = user_input()

st.subheader("Session Feature Summary")
st.dataframe(input_df, use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("🔮 Predict Purchase", use_container_width=True)
with col2:
    if st.button("🔄 Reset Inputs", use_container_width=True):
        st.rerun()

if predict_clicked:
    expected_cols = preprocessor.feature_names_in_ if hasattr(preprocessor, "feature_names_in_") else input_df.columns
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]

    X_proc = preprocessor.transform(input_df)
    proba = model.predict_proba(X_proc)[0][1]
    pred = int(proba >= 0.5)

    st.subheader("Prediction Result")
    label = "✅ Likely to Purchase" if pred == 1 else "❌ Unlikely to Purchase"
    st.markdown(f"### {label}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        title={"text": "Purchase Probability (%)"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#4C72B0"},
               "steps": [
                   {"range": [0, 40], "color": "#f8d7da"},
                   {"range": [40, 70], "color": "#fff3cd"},
                   {"range": [70, 100], "color": "#d4edda"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Model Confidence", f"{max(proba, 1-proba)*100:.1f}%")

    result_df = input_df.copy()
    result_df["Predicted_Revenue"] = pred
    result_df["Purchase_Probability"] = proba
    st.download_button("⬇️ Download Prediction", result_df.to_csv(index=False),
                        file_name="prediction.csv", mime="text/csv")

st.markdown("---")
st.markdown("<center>Built for COM763 Advanced Machine Learning — Wrexham University</center>", unsafe_allow_html=True)
