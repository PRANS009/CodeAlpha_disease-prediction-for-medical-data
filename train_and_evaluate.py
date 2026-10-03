"""
Disease Prediction from Medical Data
Task 4: Heart Disease Classification using Machine Learning
Algorithms: Logistic Regression, SVM, Random Forest, XGBoost
"""

import os
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Set seeds for reproducibility
np.random.seed(42)

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_NAMES_READABLE = {
    'age': 'Age (years)',
    'sex': 'Sex (1=Male, 0=Female)',
    'cp': 'Chest Pain Type (1:Typ Angina, 2:Atyp Angina, 3:Non-anginal, 4:Asymptomatic)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)',
    'restecg': 'Resting ECG Results (0:Normal, 1:ST-T Abnormality, 2:LV Hypertrophy)',
    'thalach': 'Max Heart Rate Achieved (bpm)',
    'exang': 'Exercise Induced Angina (1=Yes, 0=No)',
    'oldpeak': 'ST Depression Induced by Exercise',
    'slope': 'Slope of Peak Exercise ST Segment (1:Upsloping, 2:Flat, 3:Downsloping)',
    'ca': 'Major Vessels Colored by Fluoroscopy (0-3)',
    'thal': 'Thalassemia (3:Normal, 6:Fixed Defect, 7:Reversible Defect)'
}

def download_and_load_data():
    """Download the UCI Heart Disease Cleveland dataset if not already cached."""
    os.makedirs('data', exist_ok=True)
    raw_path = os.path.join('data', 'processed.cleveland.data')
    
    if not os.path.exists(raw_path):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        print(f"Downloading UCI Heart Disease dataset from {url}...")
        urllib.request.urlretrieve(url, raw_path)
        print("Download complete.")
        
    cols = FEATURE_COLUMNS + ['target']
    df = pd.read_csv(raw_path, names=cols, na_values='?')
    print(f"Initial raw dataset shape: {df.shape}")
    return df

def clean_data(df):
    """
    Step 2: Clean missing, duplicate, and incorrect values.
    Also binarize the target: 0 -> Absent (0), 1,2,3,4 -> Present (1).
    """
    print("\n--- Data Cleaning & Preprocessing ---")
    initial_len = len(df)
    
    # 1. Check & drop duplicate rows if any
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate row(s)...")
        df = df.drop_duplicates()
    else:
        print("No duplicate records found.")

    # 2. Inspect missing values
    missing = df.isnull().sum()
    print("Missing values before cleaning:")
    for col, count in missing.items():
        if count > 0:
            print(f"  - {col}: {count} missing value(s)")
            
    # Impute missing values:
    # 'ca' (major vessels 0-3) and 'thal' (3, 6, 7) are discrete clinical variables -> impute with mode
    for col in ['ca', 'thal']:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  Imputed '{col}' with mode value: {mode_val}")

    # 3. Target Binarization:
    # 0 = No presence of heart disease (< 50% diameter narrowing) -> 0 (Absent)
    # 1, 2, 3, 4 = Presence of heart disease (> 50% diameter narrowing) -> 1 (Present)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Save cleaned data to CSV
    clean_path = os.path.join('data', 'heart_disease_uci.csv')
    df.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved to: {clean_path}")
    print(f"Final dataset shape: {df.shape}")
    print("Target distribution:")
    counts = df['target'].value_counts()
    print(f"  Absent (0): {counts[0]} ({counts[0]/len(df)*100:.1f}%)")
    print(f"  Present (1): {counts[1]} ({counts[1]/len(df)*100:.1f}%)")
    
    return df

def train_and_evaluate():
    """Execute the full ML pipeline."""
    # Step 1: Collect / Load data
    df = download_and_load_data()
    
    # Step 2: Clean data
    df_clean = clean_data(df)
    
    # Step 3: Separate input features (X) and target (y)
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['target']
    
    # Step 4: Split data into training and testing sets (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size:  {X_test.shape[0]} samples (unseen data)")
    
    # Step 5: Scale the numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 6: Define and Train classification models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    roc_data = {}
    cm_dict = {}
    
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Step 7: Test on unseen test data
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_scaled)
        
        # Step 8: Evaluate using Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': round(acc * 100, 2),
            'Precision': round(prec * 100, 2),
            'Recall': round(rec * 100, 2),
            'F1_Score': round(f1 * 100, 2),
            'ROC_AUC': round(auc * 100, 2)
        }
        
        cm_dict[name] = cm
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(auc, 4)}
        
        print(f"Results for {name}:")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall:    {rec*100:.2f}%")
        print(f"  F1-Score:  {f1*100:.2f}%")
        print(f"  ROC-AUC:   {auc*100:.2f}%")
        print("  Confusion Matrix:")
        print(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
        print(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
    # Step 9: Compare models based on performance
    print("\n--- Model Performance Comparison ---")
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    
    # Identify Best Model by F1-Score & Accuracy
    best_model_name = results_df['F1_Score'].idxmax()
    print(f"\nTop Performing Model by F1-Score: {best_model_name} ({results_df.loc[best_model_name, 'F1_Score']}%)")
    
    # Save artifacts
    save_artifacts(models, scaler, results, cm_dict, roc_data, X.columns.tolist(), best_model_name)
    
    # Step 10: Demonstrate prediction for new patient data
    demonstrate_new_patient_prediction(models[best_model_name], scaler, best_model_name)
    
    return models, scaler, results

def save_artifacts(models, scaler, results, cm_dict, roc_data, feature_names, best_model_name):
    """Save trained models, scaler, metrics, and visualization plots."""
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('static', 'plots'), exist_ok=True)
    
    # 1. Save scaler and models
    joblib.dump(scaler, os.path.join('models', 'scaler.joblib'))
    
    model_filenames = {
        'Logistic Regression': 'logistic_regression.joblib',
        'SVM': 'svm.joblib',
        'Random Forest': 'random_forest.joblib',
        'XGBoost': 'xgboost.joblib'
    }
    
    for name, model in models.items():
        joblib.dump(model, os.path.join('models', model_filenames[name]))
        
    # 2. Save metrics json
    metrics_path = os.path.join('models', 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': results,
            'best_model': best_model_name,
            'features': feature_names,
            'roc_data': roc_data
        }, f, indent=4)
    print(f"\nSaved models and metrics to 'models/' directory.")
    
    # 3. Generate high-quality visualizations
    generate_visualizations(results, cm_dict, roc_data, models['Random Forest'], models['XGBoost'], feature_names)

def generate_visualizations(results, cm_dict, roc_data, rf_model, xgb_model, feature_names):
    """Generate plots for comparison, confusion matrices, ROC curves, and feature importance."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    df_metrics = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    melted = df_metrics.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score'],
                             var_name='Metric', value_name='Score (%)')
    
    palette = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    ax = sns.barplot(data=melted, x='Model', y='Score (%)', hue='Metric', palette=palette)
    plt.title('Heart Disease Classification - Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(50, 100)
    plt.ylabel('Score (%)', fontsize=11)
    plt.xlabel('')
    plt.legend(loc='lower right', frameon=True)
    
    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig('static/plots/model_comparison.png', dpi=200)
    plt.close()
    
    # Plot 2: Confusion Matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    model_names = list(cm_dict.keys())
    
    for i, name in enumerate(model_names):
        cm = cm_dict[name]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    xticklabels=['Absent (0)', 'Present (1)'],
                    yticklabels=['Absent (0)', 'Present (1)'])
        axes[i].set_title(f"{name}\n(Acc: {results[name]['Accuracy']}%, F1: {results[name]['F1_Score']}%)",
                          fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        axes[i].set_ylabel('True Label', fontsize=10)
        
    plt.suptitle('Confusion Matrices on Unseen Test Data', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('static/plots/confusion_matrices.png', dpi=200)
    plt.close()
    
    # Plot 3: Combined ROC Curves
    plt.figure(figsize=(8, 6))
    colors = ['#2563eb', '#16a34a', '#d97706', '#9333ea']
    for idx, (name, d) in enumerate(roc_data.items()):
        plt.plot(d['fpr'], d['tpr'], label=f"{name} (AUC = {d['auc']:.3f})", color=colors[idx], lw=2.2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig('static/plots/roc_curves.png', dpi=200)
    plt.close()
    
    # Plot 4: Feature Importance Comparison (Random Forest vs XGBoost)
    plt.figure(figsize=(10, 6))
    rf_imp = rf_model.feature_importances_
    xgb_imp = xgb_model.feature_importances_
    
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Random Forest': rf_imp,
        'XGBoost': xgb_imp
    }).sort_values(by='Random Forest', ascending=True)
    
    y_pos = np.arange(len(feature_names))
    width = 0.35
    
    plt.barh(y_pos - width/2, feat_df['Random Forest'], width, label='Random Forest', color='#3b82f6')
    plt.barh(y_pos + width/2, feat_df['XGBoost'], width, label='XGBoost', color='#10b981')
    
    plt.yticks(y_pos, feat_df['Feature'], fontsize=10)
    plt.xlabel('Relative Feature Importance Score', fontsize=11)
    plt.title('Feature Importance in Disease Classification (RF vs XGBoost)', fontsize=13, fontweight='bold')
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png', dpi=200)
    plt.close()
    print("Generated all visualization charts in 'static/plots/'.")

def demonstrate_new_patient_prediction(model, scaler, model_name):
    """
    Step 10: Use the selected model to predict the disease class for new patient data.
    Output: The model predicts whether the disease class is Present or Absent.
    """
    print("\n=======================================================")
    print(f"STEP 10: NEW PATIENT PREDICTION DEMONSTRATION ({model_name})")
    print("=======================================================")
    
    # Example 1: High risk patient profile
    # 67 yo male, asymptomatic chest pain (4), bp 160, chol 286, fbs 0, restecg 2, thalach 108, exang 1, oldpeak 1.5, slope 2, ca 3, thal 3
    patient_high_risk = {
        'age': 67.0, 'sex': 1.0, 'cp': 4.0, 'trestbps': 160.0, 'chol': 286.0,
        'fbs': 0.0, 'restecg': 2.0, 'thalach': 108.0, 'exang': 1.0, 'oldpeak': 1.5,
        'slope': 2.0, 'ca': 3.0, 'thal': 3.0
    }
    
    # Example 2: Low risk healthy patient profile
    # 41 yo female, atypical angina (2), bp 130, chol 204, fbs 0, restecg 2, thalach 172, exang 0, oldpeak 1.4, slope 1, ca 0, thal 3
    patient_low_risk = {
        'age': 41.0, 'sex': 0.0, 'cp': 2.0, 'trestbps': 120.0, 'chol': 180.0,
        'fbs': 0.0, 'restecg': 0.0, 'thalach': 175.0, 'exang': 0.0, 'oldpeak': 0.0,
        'slope': 1.0, 'ca': 0.0, 'thal': 3.0
    }
    
    for label, patient in [("Patient A (Elevated Risk Profile)", patient_high_risk),
                          ("Patient B (Low Risk Profile)", patient_low_risk)]:
        patient_df = pd.DataFrame([patient])[FEATURE_COLUMNS]
        patient_scaled = scaler.transform(patient_df)
        pred_class = model.predict(patient_scaled)[0]
        pred_prob = model.predict_proba(patient_scaled)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        disease_status = "Present" if pred_class == 1 else "Absent"
        confidence = pred_prob if pred_class == 1 else (1.0 - pred_prob)
        
        print(f"\n{label}:")
        print(f"  Vitals: Age={patient['age']}, Sex={'Male' if patient['sex']==1 else 'Female'}, BP={patient['trestbps']} mmHg, Chol={patient['chol']} mg/dl, MaxHR={patient['thalach']} bpm")
        print(f"  Output Disease Class: >>> {disease_status} <<<")
        print(f"  Model Confidence:     {confidence*100:.2f}%")
        print(f"  Estimated Disease Probability: {pred_prob*100:.2f}%")
        print("  Clinical Note: Educational prediction demonstration. Not for medical diagnosis.")

if __name__ == '__main__':
    train_and_evaluate()
