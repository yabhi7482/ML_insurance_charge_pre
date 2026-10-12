
# ============================================================
# MEDICAL INSURANCE COST PREDICTION
# Streamlit UI - No External CSS / No Extension Required
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("insurance.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ insurance.pkl file could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("💰 Medical Insurance Cost Predictor")

st.write(
    "### Predict your estimated medical insurance charges using Machine Learning."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("💰 Insurance Predictor")

    st.divider()

    st.subheader("🤖 Model")
    st.write("Linear Regression")

    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("R²", "0.773")

    with col2:
        st.metric("Adj. R²", "0.770")

    st.divider()

    st.subheader("📌 Features")

    st.write("• Age")
    st.write("• Gender")
    st.write("• BMI")
    st.write("• Children")
    st.write("• Smoking Status")
    st.write("• Region")
    st.write("• BMI Category")

    st.divider()

    st.caption("Built using Python + Scikit-learn + Streamlit")


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.header("👤 Personal Information")

st.write("Enter your basic personal details below.")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )


with col2:

    gender = st.selectbox(
        "⚥ Gender",
        ["Male", "Female"]
    )


with col3:

    children = st.number_input(
        "👶 Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


st.divider()


# ============================================================
# HEALTH & LIFESTYLE
# ============================================================

st.header("🏥 Health & Lifestyle")

st.write("Enter your health and lifestyle information.")

col1, col2, col3 = st.columns(3)


with col1:

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        format="%.1f"
    )


with col2:

    smoker = st.selectbox(
        "🚬 Smoking Status",
        ["No", "Yes"]
    )


with col3:

    # EXACTLY FOUR REGIONS
    region = st.selectbox(
        "📍 Region",
        [
            "Northeast",
            "Northwest",
            "Southeast",
            "Southwest"
        ]
    )


# ============================================================
# BMI CATEGORY
# ============================================================

if bmi < 18.5:

    bmi_category = "Underweight"

elif bmi < 25:

    bmi_category = "Normal"

elif bmi < 30:

    bmi_category = "Overweight"

else:

    bmi_category = "Obese"


st.info(
    f"📊 Your BMI Category: **{bmi_category}**"
)


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.header("🔮 Prediction")

predict_button = st.button(
    "💰 Predict Insurance Charges",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # GENDER ENCODING
    # --------------------------------------------------------

    female_sex = 1 if gender == "Female" else 0


    # --------------------------------------------------------
    # SMOKER ENCODING
    # --------------------------------------------------------

    smoker_yes = 1 if smoker == "Yes" else 0


    # --------------------------------------------------------
    # REGION ENCODING
    #
    # Training used:
    #
    # pd.get_dummies(
    #     df_cleaned,
    #     columns=['region'],
    #     drop_first=True
    # )
    #
    # Northeast is therefore the reference category.
    # --------------------------------------------------------

    region_northwest = 1 if region == "Northwest" else 0

    region_southeast = 1 if region == "Southeast" else 0

    region_southwest = 1 if region == "Southwest" else 0


    # --------------------------------------------------------
    # BMI CATEGORY ENCODING
    #
    # drop_first=True
    #
    # Underweight is the reference category.
    # --------------------------------------------------------

    bmi_categories_Normal = (
        1 if bmi_category == "Normal" else 0
    )

    bmi_categories_Overweight = (
        1 if bmi_category == "Overweight" else 0
    )

    bmi_categories_Obese = (
        1 if bmi_category == "Obese" else 0
    )


    # ========================================================
    # STANDARD SCALING
    # ========================================================

    # Values from the standard insurance.csv dataset

    AGE_MEAN = 39.207025
    AGE_STD = 14.049960

    BMI_MEAN = 30.663397
    BMI_STD = 6.098187

    CHILDREN_MEAN = 1.094477
    CHILDREN_STD = 1.205493


    age_scaled = (
        age - AGE_MEAN
    ) / AGE_STD


    bmi_scaled = (
        bmi - BMI_MEAN
    ) / BMI_STD


    children_scaled = (
        children - CHILDREN_MEAN
    ) / CHILDREN_STD


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame({

        "age": [age_scaled],

        "female_sex": [female_sex],

        "bmi": [bmi_scaled],

        "children": [children_scaled],

        "smoker_yes": [smoker_yes],

        "region_northwest": [
            region_northwest
        ],

        "region_southeast": [
            region_southeast
        ],

        "region_southwest": [
            region_southwest
        ],

        "bmi_categories_Normal": [
            bmi_categories_Normal
        ],

        "bmi_categories_Overweight": [
            bmi_categories_Overweight
        ],

        "bmi_categories_Obese": [
            bmi_categories_Obese
        ]
    })


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        log_prediction = model.predict(
            input_data
        )[0]

        # Training target:
        #
        # df['charges'] = np.log(df['charges'])
        #
        # Convert log prediction back to original charges.

        predicted_charges = np.exp(
            log_prediction
        )

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.code(str(e))

        st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    st.success("✅ Prediction completed successfully!")

    st.metric(
        label="💰 Estimated Medical Insurance Charges",
        value=f"₹ {predicted_charges:,.2f}"
    )


    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.divider()

    st.subheader("📋 Prediction Summary")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🎂 Age",
            f"{age} years"
        )


    with col2:

        st.metric(
            "⚖️ BMI",
            f"{bmi:.1f}"
        )


    with col3:

        st.metric(
            "👶 Children",
            str(children)
        )


    with col4:

        st.metric(
            "📍 Region",
            region
        )


    # ========================================================
    # ADDITIONAL SUMMARY
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            f"⚥ Gender: **{gender}**"
        )


    with col2:

        if smoker == "Yes":

            st.warning(
                "🚬 Smoking Status: **Yes**"
            )

        else:

            st.success(
                "🚭 Smoking Status: **No**"
            )


    with col3:

        st.info(
            f"📊 BMI Category: **{bmi_category}**"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚠️ This application is created for educational and "
    "demonstration purposes. Predictions are estimates."
)

