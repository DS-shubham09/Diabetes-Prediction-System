
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Diabetes Health Portal",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

saved_data = joblib.load("diabetes_model.pkl")

model = saved_data["model"]
imputer = saved_data["imputer"]
threshold = saved_data["threshold"]
features = saved_data["features"]
zero_columns = saved_data["zero_columns"]

# ---------------- SESSION STATE ----------------

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color: #f7f6ff;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1250px;
}

header {
    background: transparent !important;
}

h1, h2, h3 {
    color: #29245c;
}

.hero {
    background: linear-gradient(120deg, #6256c9, #a18be8);
    padding: 23px 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}

.hero h1 {
    color: white;
    font-size: 30px;
    margin: 0;
}

.hero p {
    color: #f1edff;
    margin-top: 7px;
}

.card {
    background: white;
    border: 1px solid #e8e3fa;
    border-radius: 15px;
    padding: 17px;
    margin-bottom: 12px;
}

.tip {
    background: #f0edff;
    border-radius: 12px;
    padding: 13px;
    border: 1px solid #e3ddff;
    min-height: 90px;
}

.tip b {
    color: #5145a6;
}

.result {
    background: #f1edff;
    border: 1px solid #ded5ff;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.stButton > button {
    background: #7161d5;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #5747bb;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #efecff;
}

.small {
    color: #77718f;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR NAVIGATION ----------------

st.sidebar.markdown("## 🩺 Health Portal")
st.sidebar.caption("Your health information space")

page = st.sidebar.radio(
    "Navigation",
    [
        "🩺 Diabetes Screening",
        "📊 Health Insights",
        "👨‍⚕️ Healthcare Support"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Health information and educational support")

# =====================================================
# PAGE 1: DIABETES SCREENING
# =====================================================

if page == "🩺 Diabetes Screening":

    st.markdown("""
    <div class="hero">
        <h1>Diabetes Screening</h1>
        <p>Enter your health details to receive a machine-learning estimate.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([2.1, 1], gap="medium")

    with left:

        st.markdown("### 👤 Patient Information")

        with st.form("screening_form"):

            c1, c2 = st.columns(2)

            with c1:

                pregnancies = st.number_input(
                    "Pregnancies",
                    0, 20, 1
                )

                glucose = st.number_input(
                    "Glucose (mg/dL)",
                    0.0, 300.0, 120.0
                )

                blood_pressure = st.number_input(
                    "Blood Pressure (mmHg)",
                    0.0, 200.0, 70.0
                )

                skin_thickness = st.number_input(
                    "Skin Thickness (mm)",
                    0.0, 100.0, 20.0
                )

            with c2:

                insulin = st.number_input(
                    "Insulin (mu U/ml)",
                    0.0, 900.0, 80.0
                )

                bmi = st.number_input(
                    "BMI (kg/m²)",
                    0.0, 100.0, 25.0
                )

                pedigree = st.number_input(
                    "Diabetes Pedigree Function",
                    0.0, 3.0, 0.5
                )

                age = st.number_input(
                    "Age (years)",
                    1, 120, 30
                )

            submitted = st.form_submit_button(
                "🔍 Predict",
                use_container_width=True
            )

        if submitted:

            input_data = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age
            }])

            input_data = input_data[features]

            # Handle invalid zero measurements
            input_data[zero_columns] = input_data[
                zero_columns
            ].replace(0, float("nan"))

            # Apply the saved training medians without calling imputer.transform().
            # This avoids the missing _fill_dtype attribute error when the
            # saved imputer and deployed scikit-learn versions differ.
            median_values = pd.Series(imputer.statistics_, index=features)
            input_data = input_data.fillna(median_values)

            # Prediction
            probability = model.predict_proba(input_data)[0][1]

            prediction = int(probability >= threshold)

            st.session_state.prediction = prediction
            st.session_state.probability = probability
            st.session_state.patient_data = {
                "BMI": bmi,
                "Glucose": glucose,
                "Age": age,
                "Blood Pressure": blood_pressure
            }

    with right:

        st.markdown("### 💜 Health Tips")

        st.markdown("""
        <div class="tip">
            <b>🥗 Eat Balanced</b><br>
            <span class="small">Choose whole grains, vegetables and fruits.</span>
        </div>
        <br>
        <div class="tip">
            <b>🚶 Stay Active</b><br>
            <span class="small">Include regular physical activity in your routine.</span>
        </div>
        <br>
        <div class="tip">
            <b>😴 Sleep Well</b><br>
            <span class="small">Maintain a regular and healthy sleep schedule.</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <b>💧 Remember</b>
            <p class="small">
            Drink water regularly and attend recommended health checkups.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction result

    if st.session_state.prediction is not None:

        st.markdown("---")
        st.markdown("### 📋 Screening Result")

        probability = st.session_state.probability
        prediction = st.session_state.prediction

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:

            if prediction == 1:
                st.error("Higher Rick")
            else:
                st.success("Lower Rick")

        with result_col2:

            st.metric(
                "Model-estimated probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

        st.caption(
            "This is a machine-learning estimate, not a medical diagnosis. "
            "Please consult a qualified healthcare professional."
        )

# =====================================================
# PAGE 2: HEALTH INSIGHTS
# =====================================================

elif page == "📊 Health Insights":

    st.markdown("""
    <div class="hero">
        <h1>Health Insights</h1>
        <p>Explore your screening result and useful health information.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Your Screening Overview")

    if st.session_state.probability is not None:

        probability = st.session_state.probability

        fig = go.Figure(data=[
            go.Pie(
                labels=["Estimated probability", "Remaining probability"],
                values=[
                    probability * 100,
                    100 - probability * 100
                ],
                hole=0.68,
                marker=dict(
                    colors=["#7968d8", "#e9e5f8"]
                ),
                textinfo="none"
            )
        ])

        fig.update_layout(
            annotations=[dict(
                text=f"{probability * 100:.1f}%",
                x=0.5,
                y=0.5,
                font_size=25,
                showarrow=False,
                font_color="#5145a6"
            )],
            showlegend=True,
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "The chart shows the model-estimated probability for the "
            "positive class. It is not a clinically validated risk score."
        )

        patient = st.session_state.patient_data

        m1, m2, m3 = st.columns(3)

        m1.metric("Age", f"{patient['Age']} years")
        m2.metric("BMI entered", patient["BMI"])
        m3.metric("Glucose entered", f"{patient['Glucose']} mg/dL")

    else:

        st.info(
            "Complete a screening on the Diabetes Screening page "
            "to view your result here."
        )

    st.markdown("---")
    st.markdown("### ⚖️ BMI Calculator")

    bmi_col1, bmi_col2 = st.columns(2)

    with bmi_col1:

        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=170.0
        )

    with bmi_col2:

        weight = st.number_input(
            "Weight (kg)",
            min_value=10.0,
            max_value=300.0,
            value=65.0
        )

    calculated_bmi = weight / ((height / 100) ** 2)

    st.metric("Your calculated BMI", f"{calculated_bmi:.1f}")

    if calculated_bmi < 18.5:
        st.info("BMI category: Underweight")
    elif calculated_bmi < 25:
        st.success("BMI category: Healthy weight range")
    elif calculated_bmi < 30:
        st.warning("BMI category: Overweight")
    else:
        st.warning("BMI category: Obesity range")

    st.caption(
        "BMI is a screening measure, not a diagnosis. "
        "Adult BMI categories may not apply to children or pregnancy."
    )

    st.markdown("---")
    st.markdown("### 🩸 General Glucose Information")

    st.markdown("""
    | Test | Common reference information |
    |---|---|
    | Fasting glucose | Below 100 mg/dL is generally normal |
    | Fasting glucose | 100–125 mg/dL may indicate prediabetes |
    | Fasting glucose | 126 mg/dL or above may indicate diabetes |
    | HbA1c | Below 5.7% is generally normal |
    """)

    st.caption(
        "These are general adult screening reference ranges. "
        "Diagnosis requires appropriate laboratory testing and "
        "professional interpretation."
    )

    st.markdown("---")

    st.markdown("### 💡 Daily Healthy Habits")

    habit1, habit2, habit3 = st.columns(3)

    with habit1:
        st.markdown("""
        <div class="tip">
        <b>🥗 Balanced Diet</b>
        <p class="small">Include fibre-rich foods and balanced meals.</p>
        </div>
        """, unsafe_allow_html=True)

    with habit2:
        st.markdown("""
        <div class="tip">
        <b>🏃 Physical Activity</b>
        <p class="small">Build a regular activity routine.</p>
        </div>
        """, unsafe_allow_html=True)

    with habit3:
        st.markdown("""
        <div class="tip">
        <b>🩺 Health Checkups</b>
        <p class="small">Follow recommended screening schedules.</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# PAGE 3: HEALTHCARE SUPPORT
# =====================================================

elif page == "👨‍⚕️ Healthcare Support":

    st.markdown("""
    <div class="hero">
        <h1>Healthcare Support</h1>
        <p>Find the right healthcare professional and prepare for a consultation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👨‍⚕️ Which specialist can help?")

    specialist1, specialist2 = st.columns(2)

    with specialist1:

        st.markdown("""
        <div class="card">
            <h3>🩺 Endocrinologist</h3>
            <p class="small">
            A specialist who treats diabetes and other hormone-related conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with specialist2:

        st.markdown("""
        <div class="card">
            <h3>👨‍⚕️ General Physician</h3>
            <p class="small">
            Can provide an initial health assessment and recommend further testing.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📍 Find Healthcare Support")

    st.write(
        "Enter your city to identify nearby healthcare services."
    )

    city = st.text_input(
        "Your city",
        placeholder="Example: Amritsar"
    )

    if st.button("Find Healthcare Services"):

        if city.strip():

            from urllib.parse import quote

            search_url = (
                "https://www.google.com/maps/search/"
                + quote("diabetes specialist endocrinologist " + city)
            )

            st.markdown(
                f"[Find diabetes specialists near {city}]({search_url})"
            )

        else:

            st.warning("Please enter your city.")

    st.markdown("---")

    st.markdown("### 📞 Specialist Contact Directory")

    st.info(
        "Verified specialist phone numbers will be added here "
        "once a location is selected. We do not display unverified numbers."
    )

    st.markdown("### 📝 Before Your Appointment")

    st.markdown("""
    - Keep your recent blood glucose and HbA1c reports.
    - Note any medicines you currently take.
    - Write down your symptoms and questions.
    - Share your family history with the healthcare professional.
    - Ask whether additional laboratory testing is required.
    """)

    st.warning(
        "If you experience severe or sudden symptoms, seek urgent "
        "medical care through your local emergency service. "
        "Do not rely on this application for emergencies."
    )

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown("""
<center>
<p style="color:#77718f;">
Diabetes Health Portal | Educational Health Application
</p>
<p style="color:#9993ae; font-size:12px;">
Not a medical diagnosis or substitute for professional healthcare.
</p>
</center>
""", unsafe_allow_html=True)
