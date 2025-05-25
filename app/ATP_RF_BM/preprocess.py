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
    """
    matrix = one_hot_encoding(sequence)
    window_features = []
    for i in range(len(matrix) - window_size + 1):
        window = np.array(matrix[i:i+window_size])
        window_features.append(window)
    return window_features