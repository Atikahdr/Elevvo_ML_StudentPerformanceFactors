import pickle
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime


# Load Model dan preprocessor
model = joblib.load("lr_poly.pkl")      # model LogisticRegression
poly = joblib.load("poly.pkl")           # PolynomialFeatures
preprocessor = joblib.load("preprocessor.pkl")

# 🔹 Session State Initializer
if "page" not in st.session_state:
    st.session_state.page = "home"
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "probability" not in st.session_state:
    st.session_state.probability = None
if "history" not in st.session_state:
    st.session_state.history = []

# 🔹 Configuration Page
st.set_page_config(
    page_title="Student Performance App",
    page_icon="🎓",
    layout="wide",
)

# 🔹 Page Navigation
st.sidebar.title("Student Performance")

page_map = {
    "🎓 Home Page": "home",
    "📝 Input Data": "input",
    "📋 Data Table": "table",
    "📊 Bar Chart": "barchart",
    "🧾 History": "history"
}
reverse_map = {v: k for k, v in page_map.items()}

menu = st.sidebar.radio(
    "Select Page:",
    list(page_map.keys()),
    index=list(page_map.values()).index(st.session_state.page)
)
st.session_state.page = page_map[menu]
st.sidebar.markdown("---")
st.sidebar.caption("Created by **Atikah DR**")

# 🔹 Prediction Function
def predict_student(model, preprocessor, poly, data_input):
    try:
        if data_input is None or data_input.empty:
            return None, None

        X_processed = preprocessor.transform(data_input)
        X_poly = poly.transform(X_processed)

        prediction = model.predict(X_poly)
        probability = model.predict_proba(X_poly)

        return prediction, probability
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# 🔹 HOME PAGE
if st.session_state.page == "home":
    st.title("🎓 Student Performance Prediction App")
    header_img = "Student.jpg"
    st.subheader(" **Welcome!** ")
    st.markdown("""
        This interactive application helps you predict a student's **performance category**
        using a Machine Learning classification model.
        Enter academic, lifestyle, and socio-economic factors such as study hours,
        attendance, parental involvement, motivation level, and more to see the predicted performance category.
    """)
    st.markdown(
    """
    🧠 **Model Insight:**  
    This application uses a **Polynomial Logistic Regression model**, which creates **polynomial features** from the original student data to better capture non-linear relationships between academic, lifestyle, and socio-economic factors.  
    
    📊 Model performance metrics for student classification.  
    🧪 **ROC AUC:** 98.23%  
    ✅ **Accuracy:** 91.07%  
    🎯 **Precision:** 91.12%  
    🔄 **Recall:** 91.07%  
    ⚖️ **F1 Score:** 91.09%  
    🚀 **F2 Score:** 91.08%  

    Using polynomial features allows the model to detect more complex patterns, helping provide more accurate predictions of student performance.
    """)
    st.subheader("Why use this application?")
    st.markdown("""
        📊 **Data-driven** — Built using a trained Machine Learning model.  
        ⚡ **Fast Prediction** — Get instant classification results.  
        📈 **Insightful** — Understand how student factors influence performance.  
        🎓 **Educational** — Useful for educators, analysts, and students.
    """)
    st.subheader("How to Use")
    st.markdown("""
        1. Select **Input Data** from the sidebar.  
        2. Fill in the student information.  
        3. Save your input.  
        4. Open the **Table Page** to review data and prediction results.
    """)
    if st.button("📝 Input Student Data"):
        st.session_state.page = "input"
        st.rerun()

# 🔹 INPUT DATA PAGE
elif st.session_state.page == "input":
    st.title("🎓 Student Performance Prediction")
    st.write("Fill in the student details below to predict performance category.")

    # Academic Information
    st.subheader("📘 Academic Information")
    col1, col2 = st.columns(2)
    with col1:
        Hours_Studied = st.number_input("Hours Studied", min_value=1, max_value=50, value=23, step=1)
        Previous_Scores = st.number_input("Previous Scores", min_value=50, max_value=100, value=60, step=1)
        Tutoring_Sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=8, value=3, step=1)
        Attendance = st.slider("Attendance", min_value=6, max_value=100, value=90, step=1)
    with col2:
        Motivation_Level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        Teacher_Quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
        School_Type = st.selectbox("School Type", ["Public", "Private"])
        Peer_Influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])

    st.divider()

    # Family & Resources
    st.subheader("👨‍👩‍👧 Family & Resources")
    col3, col4 = st.columns(2)
    with col3:
        Parental_Involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        Parental_Education_Level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
        Family_Income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    with col4:
        Access_to_Resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
        Internet_Access = st.selectbox("Internet Access", ["No", "Yes"])
        Distance_from_Home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

    st.divider()

    # Lifestyle
    st.subheader("🏃 Lifestyle")
    col5, col6 = st.columns(2)
    with col5:
        Learning_Disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
        Gender = st.selectbox("Gender", ["Female", "Male"])
        Extracurricular_Activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    with col6:
        Sleep_Hours = st.slider("Sleep Hours", min_value=4, max_value=12, value=8, step=1)
        Physical_Activity = st.slider("Physical Activity (hours)", min_value=0, max_value=6, value=3, step=1)

    st.divider()

    # Derived Features
    def create_features(Hours_Studied, Attendance, Tutoring_Sessions,
                        Previous_Scores, Motivation_Level,
                        Sleep_Hours, Physical_Activity):
        lm_map = {'Low': 0.8, 'Medium': 1.0, 'High': 1.2}
        motivation_risk_map = {'Low': 3, 'Medium': 2, 'High': 1}

        Academic_Effort = Hours_Studied*0.4 + Attendance*0.4 + Tutoring_Sessions*0.2
        Learning_Momentum = Previous_Scores * lm_map[Motivation_Level]
        Lifestyle_Balance = Sleep_Hours*0.6 + Physical_Activity*0.4
        Risk_Index = (100 - Attendance)*0.5 + max(0,(8-Sleep_Hours))*0.3 + motivation_risk_map[Motivation_Level]*0.2
        return Academic_Effort, Learning_Momentum, Lifestyle_Balance, Risk_Index

    Academic_Effort, Learning_Momentum, Lifestyle_Balance, Risk_Index = create_features(
        Hours_Studied, Attendance, Tutoring_Sessions, Previous_Scores,
        Motivation_Level, Sleep_Hours, Physical_Activity
    )

    col7, col8 = st.columns(2)
    with col7:
        st.metric("Academic Effort", round(Academic_Effort, 2))
        st.metric("Learning Momentum", round(Learning_Momentum, 2))
    with col8:
        st.metric("Lifestyle Balance", round(Lifestyle_Balance, 2))
        st.metric("Risk Index", round(Risk_Index, 2))

    st.divider()

    # 🔹 SAVE INPUT & PREDICT
    if st.button("Save Input Data"):
        input_df = pd.DataFrame([{
            "Hours_Studied": Hours_Studied,
            "Attendance": Attendance,
            "Sleep_Hours": Sleep_Hours,
            "Previous_Scores": Previous_Scores,
            "Tutoring_Sessions": Tutoring_Sessions,
            "Physical_Activity": Physical_Activity,
            "Extracurricular_Activities": Extracurricular_Activities,
            "Learning_Disabilities": Learning_Disabilities,
            "Parental_Involvement": Parental_Involvement,
            "Access_to_Resources": Access_to_Resources,
            "Motivation_Level": Motivation_Level,
            "Internet_Access": Internet_Access,
            "Family_Income": Family_Income,
            "Teacher_Quality": Teacher_Quality,
            "School_Type": School_Type,
            "Peer_Influence": Peer_Influence,
            "Parental_Education_Level": Parental_Education_Level,
            "Distance_from_Home": Distance_from_Home,
            "Gender": Gender,
            "Academic_Effort": Academic_Effort,
            "Learning_Momentum": Learning_Momentum,
            "Lifestyle_Balance": Lifestyle_Balance,
            "Risk_Index": Risk_Index
        }])

        st.session_state.input_data = input_df

        # 🔹 Predict with try/except
        prediction, probability = predict_student(model, preprocessor, poly, input_df)
        if prediction is not None:
            st.session_state.prediction = prediction[0]
            st.session_state.probability = probability[0] if probability is not None else None
            st.success("✔ Data saved & predicted successfully!")

        # Save history
        confidence = None
        if st.session_state.probability is not None:
            confidence = round(np.max(st.session_state.probability)*100,2)

        history_row = st.session_state.input_data.copy()
        history_row["Predicted_Category"] = st.session_state.prediction
        history_row["Confidence (%)"] = confidence
        history_row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append(history_row.iloc[0].to_dict())
        st.session_state.page = "table"
        st.rerun()

# 📋 Table Session
elif st.session_state.page == "table":

    st.title("Student Performance Result")

    if st.session_state.input_data is None:
        st.warning("⚠ Please enter student data first in INPUT PAGE.")
    
    else:
        st.subheader("📝 Student Input Summary")

        data_t = st.session_state.input_data.T.astype(str)
        data_t.columns = ["Value"]
        st.dataframe(data_t, use_container_width=True)

        st.divider()

        # Prediction 
        try:
            prediction, prob = predict_student(
                model,
                preprocessor,
                poly,
                st.session_state.input_data
            )

            if prediction is not None:

                predicted_label = prediction[0]
                probabilities = prob[0]
                confidence = np.max(probabilities) * 100

                st.subheader("🎯 Prediction Result")

                if predicted_label == "Low":
                    st.warning("⚠️ Low Performance Student")
                elif predicted_label == "Medium":
                    st.info("📘 Medium Performance Student")
                else:  # High
                    st.success("🌟 High Performance Student")

            else:
                st.error("Prediction failed. Please check your inputs.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

        st.markdown(
        """
        The prediction is made using a **Polynomial Logistic Regression model**.  
        """ )
# 📊 Bar Chart Session (Polynomial)
elif st.session_state.page == "barchart":

    st.header("📊 Probability Class Prediction")

    if st.session_state.input_data is None:
        st.warning("⚠ Please enter student data first in INPUT PAGE.")
    
    else:
        data_input = st.session_state.input_data

        try:
            #Preprocess + Polynomial Features
            X_processed = preprocessor.transform(data_input)
            X_poly = poly.transform(X_processed)
            prediction = model.predict(X_poly)[0]
            probabilities = model.predict_proba(X_poly)[0]

            # Label mapping
            class_labels = [f"{c} Performance" for c in model.classes_]

            prob_df = pd.DataFrame({
                "Performance Category": class_labels,
                "Probability": probabilities
            })

            # Convert ke percent
            prob_df["Probability (%)"] = prob_df["Probability"] * 100

            # 🎓 Student Key Metrics
            st.subheader("📘 Student Key Indicators")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Hours Studied", data_input["Hours_Studied"][0])
            with col2:
                st.metric("Attendance", data_input["Attendance"][0])
            with col3:
                st.metric("Previous Scores", data_input["Previous_Scores"][0])
            with col4:
                st.metric("Academic Effort", round(data_input["Academic_Effort"][0], 2))

            st.divider()

            # 📊 Plotly Bar Chart
            st.subheader("📊 Class Probability Distribution")

            fig = px.bar(
                prob_df,
                x="Performance Category",
                y="Probability (%)",
                text="Probability (%)"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Performance Category",
                yaxis_title="Probability (%)",
                xaxis_tickangle=0,
                yaxis=dict(range=[0, 100]),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            #Prediction Result
            prediction, prob = predict_student(model, preprocessor, poly, data_input)

            if prediction is not None:
                predicted_label = prediction[0]
                confidence = np.max(prob[0]) * 100 if prob is not None else None

                st.subheader("🎯 Prediction Result")

                if predicted_label == "Low":
                    st.warning("⚠️ Low Performance Student")
                elif predicted_label == "Medium":
                    st.info("📘 Medium Performance Student")
                else:  # High
                    st.success("🌟 High Performance Student")

                if confidence is not None:
                    st.caption(f"Confidence: {confidence:.2f}%")

            else:
                st.error("Prediction failed. Please check your inputs.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")


# History Session
elif st.session_state.page == "history":

    st.title("🧾 Student Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No prediction history available.")
    
    else:
        history_df = pd.DataFrame(st.session_state.history)

        # Format confidence jika ada
        if "Confidence (%)" in history_df.columns:
            history_df["Confidence (%)"] = history_df["Confidence (%)"].apply(
                lambda x: f"{x:.2f}%" if pd.notnull(x) else "-"
            )

        # Show latest first
        st.dataframe(
            history_df.iloc[::-1],
            use_container_width=True
        )

        st.divider()

        # Summary
        if "Predicted_Category" in history_df.columns:

            total = len(history_df)
            high = (history_df["Predicted_Category"] == "High").sum()
            medium = (history_df["Predicted_Category"] == "Medium").sum()
            low = (history_df["Predicted_Category"] == "Low").sum()

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Predictions", total)
            col2.metric("High Performance", high)
            col3.metric("Medium / Low", medium + low)

        st.divider()

        # Download CSV
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download History",
            csv,
            "student_prediction_history.csv",
            "text/csv"
        )

    if st.button("🗑️ Delete History"):
        st.session_state.history = []
        st.success("History deleted successfully!")
        st.rerun()

#  Footer
st.markdown("---")


st.caption("💡Polynomial Logistic Regression | Machine Learning Prediction Project")
