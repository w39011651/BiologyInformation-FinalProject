import joblib
import os

def load_rf_model(model_path = 'rf_with_pssm_model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    return joblib.load(model_path)