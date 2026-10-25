import streamlit as st
import pandas as pd
import joblib

st.title("Iris Flower Prediction App")
st.write("Enter flower measurements to predict its species.")

model = joblib.load("iris_model.joblib")

sl = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.1)
sw = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.5)
pl = st.slider("Petal Length (cm)", 1.0, 6.9, 1.4)
pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict Flower"):
    data = pd.DataFrame(
        [[sl, sw, pl, pw]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(data)[0]
    names = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Flower: {names[prediction]}")
    probabilities = model.predict_proba(data)[0]
    st.write("Prediction Confidence:", f"{max(probabilities)*100:.2f}%")
    st.bar_chart(pd.DataFrame({
        "Species": names,
        "Probability": probabilities
    }).set_index("Species"))
