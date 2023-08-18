import pandas as pd
import streamlit as st
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Function to load the models
forest_model = joblib.load('forest_model.pkl')    

    
    return logis_model, forest_model, gumodel, dtree_model

def main():
    # Title of the web app
    st.title("Model Predictions App")
    st.write("Enter the following features to get predictions:")

    # Load the models
    logis_model, forest_model, gumodel, dtree_model = load_models()

    # User input for features
    st.header('Feature Input')

    # Create input fields for each feature
    # For numerical features, use st.number_input()
    feature1 = st.number_input("SMOKING", value=0)
    feature2 = st.number_input("PEER_PRESSURE", value=0)
    feature3 = st.number_input("ALCOHOL CONSUMING", value=0)
    feature4 = st.number_input("CHEST PAIN", value=0)
    feature5 = st.number_input("CHRONIC DISEASE", value=0)
   
    
    # Selection box for the model to use
    selected_model = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "Gaussian Naive Bayes", "Decision Tree"])

    # Button for predictions
    clicked = st.button('Get Predictions')

    # Perform predictions when the button is clicked
    if clicked:
        if selected_model == "Logistic Regression":
            model = logis_model
        elif selected_model == "Random Forest":
            model = forest_model
        elif selected_model == "Gaussian Naive Bayes":
            model = gumodel
        else:
            model = dtree_model

        # Perform predictions using the selected model
        input_features = pd.DataFrame({
            "SMOKING":[feature1],
            "PEER_PRESSURE": [feature2],
            "ALCOHOL CONSUMING": [feature3],
            "CHEST PAIN": [feature4],
            "CHRONIC DISEASE":[feature5]
        })
        prediction = forest_model.predict(input_features)

        # Display the prediction result
        st.header('Prediction')
        st.write(f'The prediction result is: {prediction[0]}')

if __name__ == '__main__':
    main()
