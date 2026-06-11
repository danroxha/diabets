import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1. Load the model and scaler
print("Loading model and scaler...")
model = tf.keras.models.load_model('diabetes_binary_mlp_model.keras')
scaler = joblib.load('scaler.joblib')

def predict_diabetes_risk(data):
    """
    Expects a dictionary or list of feature values in the correct order.
    Order: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, 
           PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, 
           GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
    """
    # Convert to DataFrame to handle feature engineering
    features = pd.DataFrame([data])
    
    # Required order (ensure it matches train.py)
    cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    
    features = features[cols]
    
    # Feature Engineering (must match train_binary.py)
    features['BMI_Age'] = features['BMI'] * features['Age']
    features['HighBP_HighChol'] = features['HighBP'] * features['HighChol']
    features['GenHlth_PhysHlth'] = features['GenHlth'] * features['PhysHlth']
    
    # Scaling
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled, verbose=0)
    probability = float(prediction[0][0])
    
    return probability

# Example usage
sample_data = {
    'HighBP': 1.0, 'HighChol': 1.0, 'CholCheck': 1.0, 'BMI': 30.0, 'Smoker': 0.0,
    'Stroke': 0.0, 'HeartDiseaseorAttack': 0.0, 'PhysActivity': 1.0, 'Fruits': 1.0,
    'Veggies': 1.0, 'HvyAlcoholConsump': 0.0, 'AnyHealthcare': 1.0, 'NoDocbcCost': 0.0,
    'GenHlth': 3.0, 'MentHlth': 0.0, 'PhysHlth': 0.0, 'DiffWalk': 0.0, 'Sex': 1.0,
    'Age': 9.0, 'Education': 5.0, 'Income': 6.0
}

risk_prob = predict_diabetes_risk(sample_data)
print(f"\nProbability of Diabetes Risk: {risk_prob*100:.2f}%")
if risk_prob > 0.5:
    print("High Risk Detected")
else:
    print("Low Risk Detected")
