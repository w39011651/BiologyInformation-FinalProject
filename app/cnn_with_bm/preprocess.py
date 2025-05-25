import numpy as np
import torch

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def one_hot_encoding(sequence):
    binary_matrix = []
    for aa in sequence:
        encoding = [0] * len(AMINO_ACIDS)
        if aa in AMINO_ACIDS:
            encoding[AMINO_ACIDS.index(aa)] = 1
        binary_matrix.append(encoding)
    return binary_matrix

def sliding_window_for_matrix(matrix, window_size):
    ret = []
    left = 0
    right = window_size
    while right <= len(matrix):
        ret.append(matrix[left:right])
        left += 1
        right += 1
    return ret

def process_single_protein(sequence, window_size=15):
    """
    sequence: str, 純胺基酸序列
    """
    bm = one_hot_encoding(sequence)
    matrix_windows = sliding_window_for_matrix(bm, window_size)
    window_tensor = torch.tensor(matrix_windows, dtype=torch.float32)  # (num_windows, 15, 20)
    return window_tensor, np.array(bm)