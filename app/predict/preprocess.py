import numpy as np

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def one_hot_encoding(sequence):
    binary_matrix = []
    for aa in sequence:
        encoding = [0] * len(AMINO_ACIDS)
        if aa in AMINO_ACIDS:
            encoding[AMINO_ACIDS.index(aa)] = 1
        binary_matrix.append(encoding)
    return binary_matrix

def sequence_to_sliding_windows(sequence, window_size=15):
    """Returns (window_features, center_indices)
    window_features: List of np.array, each shape (window_size, 20)
    center_indices: 原始序列中心點index（0-based）
    """
    matrix = one_hot_encoding(sequence)
    window_features = []
    center_indices = []
    for i in range(len(matrix) - window_size + 1):
        window = np.array(matrix[i:i+window_size])
        center_indices.append(i + window_size // 2)
        window_features.append(window)
    return window_features, center_indices