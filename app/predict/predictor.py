import numpy as np
import joblib
from .preprocess import sequence_to_sliding_windows
from .model_loader import load_rf_model

def predict_binding_sites(sequence, model_path, window_size=15):
    """
    sequence: str, 單條蛋白質序列
    model_path: str, RF模型的pkl路徑
    """
    # 特徵處理
    window_features, center_indices = sequence_to_sliding_windows(sequence, window_size=window_size)
    if len(window_features) == 0:
        print("序列長度太短無法做sliding window")
        return []
    # 扁平成RF輸入格式
    X = np.array([w.flatten() for w in window_features])
    model = load_rf_model(model_path)
    y_pred = model.predict(X)  # 0/1
    # 收集所有預測為1的中心index
    binding_site_indices = [center_indices[i] for i, pred in enumerate(y_pred) if pred == 1]
    print("Binding sites (index):", binding_site_indices)
    return binding_site_indices