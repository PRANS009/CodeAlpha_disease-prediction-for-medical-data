"""
CLI Prediction Tool for Disease Prediction from Medical Data
Task 4: Step 10 - New Patient Prediction
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

MODELS = {
    'rf': ('random_forest.joblib', 'Random Forest'),
    'xgb': ('xgboost.joblib', 'XGBoost'),
    'lr': ('logistic_regression.joblib', 'Logistic Regression'),
    'svm': ('svm.joblib', 'Support Vector Machine')
}

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

def load_resources(model_key='rf'):
    if not os.path.exists(SCALER_PATH):
        print("Error: Models and scaler not found. Please run 'python3 train_and_evaluate.py' first.")
        sys.exit(1)
        
    scaler = joblib.load(SCALER_PATH)
    model_filename, model_name = MODELS.get(model_key, MODELS['rf'])
    model_path = os.path.join(MODEL_DIR, model_filename)
    model = joblib.load(model_path)
    return scaler, model, model_name

def predict_patient(patient_data, model_key='rf'):
    scaler, model, model_name = load_resources(model_key)
    
    # Ensure DataFrame format
    df = pd.DataFrame([patient_data])[FEATURE_COLUMNS]
    scaled_features = scaler.transform(df)
    
    pred_class = model.predict(scaled_features)[0]
    prob_present = model.predict_proba(scaled_features)[0][1] if hasattr(model, 'predict_proba') else None
    
    status = "Present" if pred_class == 1 else "Absent"
    confidence = (prob_present if pred_class == 1 else (1.0 - prob_present)) * 100 if prob_present is not None else 100.0
    
    # Risk factor diagnostics
    risk_factors = []
    if patient_data['chol'] > 240:
        risk_factors.append(f"High Cholesterol ({patient_data['chol']} mg/dl > 240)")
    if patient_data['trestbps'] > 130:
        risk_factors.append(f"Elevated Blood Pressure ({patient_data['trestbps']} mmHg > 130)")
    if patient_data['thalach'] < 120:
        risk_factors.append(f"Low Maximum Heart Rate ({patient_data['thalach']} bpm)")
    if patient_data['oldpeak'] >= 2.0:
        risk_factors.append(f"Significant ST Depression ({patient_data['oldpeak']})")
    if patient_data['ca'] > 0:
        risk_factors.append(f"Major Fluoroscopy Vessels Colored ({int(patient_data['ca'])} vessels)")
    if patient_data['exang'] == 1:
        risk_factors.append("Exercise-Induced Angina reported")

    return {
        'status': status,
        'pred_class': int(pred_class),
        'probability_present': round(prob_present * 100, 2) if prob_present is not None else None,
        'confidence': round(confidence, 2),
        'model_name': model_name,
        'risk_factors': risk_factors
    }

def print_report(patient_data, result):
    print("\n=======================================================")
    print("      HEART DISEASE PREDICTION - PATIENT REPORT       ")
    print("=======================================================")
    print(f"Algorithm Used: {result['model_name']}")
    print("-" * 55)
    print("PATIENT MEDICAL PARAMETERS:")
    print(f"  • Age: {patient_data['age']:.0f} years | Sex: {'Male' if patient_data['sex']==1 else 'Female'}")
    print(f"  • Resting Blood Pressure: {patient_data['trestbps']:.0f} mm Hg")
    print(f"  • Serum Cholesterol: {patient_data['chol']:.0f} mg/dl")
    print(f"  • Fasting Blood Sugar > 120 mg/dl: {'Yes' if patient_data['fbs']==1 else 'No'}")
    print(f"  • Chest Pain Type: {int(patient_data['cp'])} (1:Typ Angina, 2:Atyp Angina, 3:Non-anginal, 4:Asymptomatic)")
    print(f"  • Resting ECG: {int(patient_data['restecg'])}")
    print(f"  • Max Heart Rate Achieved: {patient_data['thalach']:.0f} bpm")
    print(f"  • Exercise-Induced Angina: {'Yes' if patient_data['exang']==1 else 'No'}")
    print(f"  • ST Depression (oldpeak): {patient_data['oldpeak']:.1f}")
    print(f"  • ST Slope: {int(patient_data['slope'])}")
    print(f"  • Fluoroscopy Major Vessels: {int(patient_data['ca'])}")
    print(f"  • Thalassemia: {int(patient_data['thal'])}")
    print("-" * 55)
    print("PREDICTION RESULT:")
    if result['status'] == 'Present':
        print(f"  >>> DISEASE CLASS: [ PRESENT ] <<<")
        print(f"  Estimated Disease Probability: {result['probability_present']}%")
        print(f"  Confidence: {result['confidence']}%")
    else:
        print(f"  >>> DISEASE CLASS: [ ABSENT ] <<<")
        print(f"  Estimated Disease Probability: {result['probability_present']}%")
        print(f"  Confidence: {result['confidence']}%")
    
    if result['risk_factors']:
        print("\nNOTABLE CLINICAL RISK INDICATORS:")
        for rf in result['risk_factors']:
            print(f"  [!] {rf}")
    else:
        print("\nNOTABLE CLINICAL RISK INDICATORS: None flagged as elevated.")
        
    print("-" * 55)
    print("DISCLAIMER: Educational machine learning demonstration.")
    print("This is not a certified medical diagnosis system.")
    print("=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Heart Disease Prediction from Patient Medical Data")
    parser.add_argument('--model', choices=['rf', 'xgb', 'lr', 'svm'], default='rf', help="Algorithm to use")
    parser.add_argument('--age', type=float, default=None)
    parser.add_argument('--sex', type=float, default=None, help="1=Male, 0=Female")
    parser.add_argument('--cp', type=float, default=None, help="Chest pain type 1-4")
    parser.add_argument('--trestbps', type=float, default=None, help="Resting BP in mm Hg")
    parser.add_argument('--chol', type=float, default=None, help="Serum cholesterol in mg/dl")
    parser.add_argument('--fbs', type=float, default=None, help="Fasting blood sugar > 120 (1 or 0)")
    parser.add_argument('--restecg', type=float, default=None, help="Resting ECG 0, 1, or 2")
    parser.add_argument('--thalach', type=float, default=None, help="Max heart rate bpm")
    parser.add_argument('--exang', type=float, default=None, help="Exercise angina 1 or 0")
    parser.add_argument('--oldpeak', type=float, default=None, help="ST depression")
    parser.add_argument('--slope', type=float, default=None, help="ST slope 1, 2, or 3")
    parser.add_argument('--ca', type=float, default=None, help="Major vessels 0-3")
    parser.add_argument('--thal', type=float, default=None, help="Thalassemia 3, 6, or 7")
    parser.add_argument('--sample', choices=['high', 'low', 'borderline'], default=None, help="Use preset patient profile")
    
    args = parser.parse_args()
    
    sample_patients = {
        'high': {
            'age': 67.0, 'sex': 1.0, 'cp': 4.0, 'trestbps': 160.0, 'chol': 286.0,
            'fbs': 0.0, 'restecg': 2.0, 'thalach': 108.0, 'exang': 1.0, 'oldpeak': 1.5,
            'slope': 2.0, 'ca': 3.0, 'thal': 3.0
        },
        'low': {
            'age': 41.0, 'sex': 0.0, 'cp': 2.0, 'trestbps': 120.0, 'chol': 180.0,
            'fbs': 0.0, 'restecg': 0.0, 'thalach': 175.0, 'exang': 0.0, 'oldpeak': 0.0,
            'slope': 1.0, 'ca': 0.0, 'thal': 3.0
        },
        'borderline': {
            'age': 55.0, 'sex': 1.0, 'cp': 3.0, 'trestbps': 135.0, 'chol': 245.0,
            'fbs': 1.0, 'restecg': 1.0, 'thalach': 140.0, 'exang': 0.0, 'oldpeak': 1.0,
            'slope': 2.0, 'ca': 1.0, 'thal': 6.0
        }
    }
    
    if args.sample:
        patient = sample_patients[args.sample]
    elif args.age is not None:
        patient = {
            'age': args.age, 'sex': args.sex if args.sex is not None else 1.0,
            'cp': args.cp if args.cp is not None else 1.0,
            'trestbps': args.trestbps if args.trestbps is not None else 120.0,
            'chol': args.chol if args.chol is not None else 200.0,
            'fbs': args.fbs if args.fbs is not None else 0.0,
            'restecg': args.restecg if args.restecg is not None else 0.0,
            'thalach': args.thalach if args.thalach is not None else 150.0,
            'exang': args.exang if args.exang is not None else 0.0,
            'oldpeak': args.oldpeak if args.oldpeak is not None else 0.0,
            'slope': args.slope if args.slope is not None else 1.0,
            'ca': args.ca if args.ca is not None else 0.0,
            'thal': args.thal if args.thal is not None else 3.0
        }
    else:
        print("No patient arguments provided. Running demonstration on high-risk and low-risk test patients:")
        res_high = predict_patient(sample_patients['high'], args.model)
        print_report(sample_patients['high'], res_high)
        
        res_low = predict_patient(sample_patients['low'], args.model)
        print_report(sample_patients['low'], res_low)
        return
        
    result = predict_patient(patient, args.model)
    print_report(patient, result)

if __name__ == '__main__':
    main()
