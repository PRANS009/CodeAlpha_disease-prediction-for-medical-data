"""
Flask Application for Heart Disease Prediction
Task 4: Medical Data Classification
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Load scaler, models, and metrics
scaler = None
models = {}
metrics_data = {}

def load_all_models():
    global scaler, models, metrics_data
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        
    model_files = {
        'rf': ('random_forest.joblib', 'Random Forest'),
        'xgb': ('xgboost.joblib', 'XGBoost'),
        'lr': ('logistic_regression.joblib', 'Logistic Regression'),
        'svm': ('svm.joblib', 'Support Vector Machine')
    }
    
    for key, (fname, display_name) in model_files.items():
        fpath = os.path.join(MODEL_DIR, fname)
        if os.path.exists(fpath):
            models[key] = {
                'model': joblib.load(fpath),
                'name': display_name
            }
            
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)

load_all_models()

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics_data.get('metrics', {}), best_model=metrics_data.get('best_model', 'Random Forest'))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        model_choice = data.get('model', 'all')
        
        # Extract features
        patient_features = {}
        for col in FEATURE_COLUMNS:
            if col not in data:
                return jsonify({'error': f'Missing feature: {col}'}), 400
            patient_features[col] = float(data[col])
            
        df = pd.DataFrame([patient_features])[FEATURE_COLUMNS]
        scaled = scaler.transform(df)
        
        # Clinical Risk Factors Evaluation
        risk_factors = []
        if patient_features['chol'] >= 240:
            risk_factors.append(f"High Serum Cholesterol ({patient_features['chol']:.0f} mg/dl - High Risk >= 240)")
        elif patient_features['chol'] >= 200:
            risk_factors.append(f"Borderline Cholesterol ({patient_features['chol']:.0f} mg/dl)")
            
        if patient_features['trestbps'] >= 140:
            risk_factors.append(f"Stage 2 Hypertension ({patient_features['trestbps']:.0f} mm Hg)")
        elif patient_features['trestbps'] >= 130:
            risk_factors.append(f"Stage 1 Hypertension ({patient_features['trestbps']:.0f} mm Hg)")
            
        if patient_features['thalach'] < 120:
            risk_factors.append(f"Subnormal Max Heart Rate ({patient_features['thalach']:.0f} bpm)")
            
        if patient_features['exang'] == 1:
            risk_factors.append("Exercise Induced Angina present")
            
        if patient_features['oldpeak'] >= 2.0:
            risk_factors.append(f"Severe ST Depression ({patient_features['oldpeak']:.1f} mm)")
        elif patient_features['oldpeak'] >= 1.0:
            risk_factors.append(f"Moderate ST Depression ({patient_features['oldpeak']:.1f} mm)")
            
        if patient_features['ca'] > 0:
            risk_factors.append(f"{int(patient_features['ca'])} Major Vessel(s) Flourosopy Defect")
            
        if patient_features['thal'] in [6.0, 7.0]:
            thal_str = "Fixed Defect" if patient_features['thal'] == 6.0 else "Reversible Defect"
            risk_factors.append(f"Thalassemia: {thal_str}")
            
        if patient_features['cp'] == 4.0:
            risk_factors.append("Asymptomatic Chest Pain (Often associated with silent ischemia)")

        # Run predictions
        predictions = {}
        for key, item in models.items():
            mod = item['model']
            p_class = int(mod.predict(scaled)[0])
            p_prob = float(mod.predict_proba(scaled)[0][1]) if hasattr(mod, 'predict_proba') else (0.9 if p_class == 1 else 0.1)
            predictions[key] = {
                'name': item['name'],
                'status': 'Present' if p_class == 1 else 'Absent',
                'probability': round(p_prob * 100, 1),
                'confidence': round((p_prob if p_class == 1 else (1.0 - p_prob)) * 100, 1)
            }
            
        # Consensus
        present_count = sum(1 for p in predictions.values() if p['status'] == 'Present')
        avg_prob = np.mean([p['probability'] for p in predictions.values()])
        consensus_status = 'Present' if present_count >= 2 else 'Absent'
        
        # Primary response
        primary_key = 'rf' if model_choice not in models else model_choice
        primary_result = predictions[primary_key] if model_choice != 'all' else {
            'name': 'Multi-Model Consensus (Ensemble)',
            'status': consensus_status,
            'probability': round(avg_prob, 1),
            'confidence': round((avg_prob if consensus_status == 'Present' else (100.0 - avg_prob)), 1)
        }
        
        return jsonify({
            'primary': primary_result,
            'all_models': predictions,
            'risk_factors': risk_factors,
            'patient_summary': {
                'age': int(patient_features['age']),
                'sex': 'Male' if patient_features['sex'] == 1 else 'Female',
                'bp': int(patient_features['trestbps']),
                'chol': int(patient_features['chol']),
                'max_hr': int(patient_features['thalach'])
            },
            'disclaimer': 'This tool is built for educational machine learning demonstration and must not replace professional clinical evaluation.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics')
def get_metrics():
    return jsonify(metrics_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
