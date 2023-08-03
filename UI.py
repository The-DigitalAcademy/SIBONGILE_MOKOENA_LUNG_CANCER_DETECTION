import streamlit as st
import joblib

# Function to load the models
def load_models():
    forest_model_path = '/Users/da_m1_18/Downloads/LUNGS/forest_model.pkl'
    logis_model_path = '/Users/da_m1_18/Downloads/LUNGS/logis_model.pkl'
    
    forest_model = joblib.load(forest_model_path)    
    logis_model = joblib.load(logis_model_path)
    
    return logis_model, forest_model

def main():
    # Title of the web app
    st.title("Model Predictions App")
    st.write("Enter the following features to get predictions:")

    # Load the models
    logis_model, forest_model = load_models()

    # User input for features
    st.header('Feature Input')

    # Create input fields for each feature
    # For numerical features, use st.number_input()
    feature1 = st.number_input("SMOKING", value =0)
    feature2 = st.number_input("PEER_PRESSURE", value = 0)
    feature3 = st.number_input("ALCOHOL CONSUMING", value = 0)
    feature4 = st.number_input("CHEST PAIN", value=0)
    # Selection box for the model to use
    selected_model = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest"])

    # Button for predictions
    clicked = st.button('Get Predictions')

    # Perform predictions when the button is clicked
    if clicked:
        if selected_model == "Logistic Regression":
            model = logis_model
        else:
            model = forest_model

        # Perform predictions using the selected model
        prediction = model.predict([[feature1, feature2, feature3, feature4]])

        # Display the prediction result
        st.header('Prediction')
        st.write(f'The prediction result is: {prediction[0]}')

if __name__ == '__main__':
    main()

