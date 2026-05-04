import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page config (NEW - improvement 6 also included because it's standard)
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

data = joblib.load("crop_system.pkl")
model = data["model"]
le = data["encoder"]

feature_order = ['N','P','K','temperature','humidity','ph','rainfall']

def recommend_top3(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_order)

    probs = model.predict_proba(input_df)[0]
    top_idx = np.argsort(probs)[::-1][:3]

    results = []
    for i in top_idx:
        crop = le.inverse_transform([i])[0]
        prob = probs[i]
        results.append((crop, prob))

    return results


# UI HEADER (Improvement 3)
st.title("🌾 Crop Recommendation System")
st.markdown("### 🌱 Predict the best crop based on soil and climate conditions")

st.divider()

st.subheader("Enter Soil & Climate Conditions")

N = st.number_input("Nitrogen (N)", 0, 150, 90)
P = st.number_input("Phosphorus (P)", 0, 150, 42)
K = st.number_input("Potassium (K)", 0, 200, 43)
temp = st.number_input("Temperature (°C)", 0.0, 50.0, 22.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

st.divider()

if st.button("🌾 Predict Crop"):

    results = recommend_top3([N, P, K, temp, humidity, ph, rainfall])

    best_crop, best_prob = results[0]

    # MAIN RESULT
    st.success(f"🌾 Best Crop: {best_crop} ({best_prob:.2%})")

    # Confidence indicator (Improvement 4)
    if best_prob > 0.7:
        st.info("High confidence prediction ✅")
    elif best_prob > 0.4:
        st.warning("Medium confidence ⚠️")
    else:
        st.error("Low confidence ❌")

    st.divider()

    st.subheader("🌱 Top Crop Recommendations")

    # Improved display (Improvement 5)
    for i, (crop, prob) in enumerate(results, 1):
        st.write(f"**{i}. {crop} — {prob:.2%}**")
        st.progress(float(prob))
        st.markdown("---")