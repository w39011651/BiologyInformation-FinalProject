import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np   

def get_binding_site(features):
  binding_site = []
  for feature in features:
    if feature['type'] == 'Binding site' and feature['ligand']['name']=='ATP':#only need ATP
      binding_site.append([feature['location']['start']['value'], feature['location']['end']['value']])
  return binding_site

def generate_label(protein, positions):
  n = protein['sequence']['length']
  labels = [0 for _ in range(n)]

  for bind_site in positions:
    for i in range(bind_site[0], bind_site[1]+2):
      labels[i-1] = 1
  return labels

def sliding_window_for_labels(labels, window_size):
    ret = []
    left = 0
    right = window_size
    while right <= len(labels):
        ret.append(labels[(left+right)//2])
        left+=1
        right+=1
    return ret

def sliding_window_for_matrix(matrix, window_size):
    ret = []
    left = 0
    right = window_size
    while right <= len(matrix):
        ret.append(matrix[left:right])
        left+=1
        right+=1
    return ret

def sliding_window(matrix, labels, window_size = 15):
    label_windows = sliding_window_for_labels(labels, window_size = window_size)

    matrix_windows = sliding_window_for_matrix(matrix, window_size = window_size)
    return {"matrix":matrix_windows, "label":label_windows}

def one_hot_encoding(sequence):
  binary_matrix = []
  amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
  for aa in sequence:
    encoding = [0] * len(amino_acids)
    if aa in amino_acids:
      encoding[amino_acids.index(aa)] = 1
    binary_matrix.append(encoding)
  return binary_matrix

class ATPBindingDataset_BM(Dataset):
    def __init__(self, proteins_data):
        super().__init__()
        matrices, labels = self.__convert_to_submatrix__(proteins_data)
        self.matrices = matrices
        self.labels = labels

    def __convert_to_submatrix__(self, proteins_data):
        ret_matrics = []
        ret_labels = []
        for protein in tqdm(proteins_data):
            primaryAccession = protein['primaryAccession']
            feature = protein['features']
            pos = get_binding_site(feature)
            labels = generate_label(protein, pos)

            #matrix = get_pssm(f'drive/MyDrive/bio-final/sequences/{primaryAccession}.txt')
            matrix = one_hot_encoding(protein['sequence']['value'])

            ret = sliding_window(matrix, labels)

            if not len(ret['matrix']) == len(ret['label']):
                raise ValueError(f"The shape is not equal\nthe shape of matrix:{len(ret['matrix'])}\nthe shape of label:{len(ret['label'])}")
            ret_matrics += ret['matrix']
            ret_labels += ret['label']
        return ret_matrics, ret_labels

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, index):
        window_matrix = self.matrices[index]
        label = self.labels[index]

        window_matrix_tensor = torch.tensor(window_matrix, dtype=torch.float32)
        # 您的 PSSM 矩陣是 (Height, Width) = (15, 20)
        # 如果將其視為單通道圖像，需要增加一個通道維度 (Channels=1)
        # 使用 unsqueeze(0) 在第 0 個維度 (索引為 0) 增加一個維度
        window_matrix_tensor = window_matrix_tensor.unsqueeze(0) # 形狀變為 (1, 15, 20)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return window_matrix_tensor, label_tensor