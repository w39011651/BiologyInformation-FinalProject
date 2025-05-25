import ATP_msCNN.ATP_Dataset as ATP_Dataset
import torch

def process_single_protein(protein_id, file_folder):
    pssm = ATP_Dataset.get_pssm(f'{file_folder}/{protein_id}.txt')#or txt
    matrix_windows = ATP_Dataset.sliding_window_for_matrix(pssm, 15)
    window_tensor = torch.tensor(matrix_windows, dtype=torch.float32)
    return window_tensor, pssm
