import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from ATP_XGB_PSSM.preprocess import process_single_protein

def predict(primary_accession, model):
    pssm_submatrix_list= process_single_protein(primary_accession, '.')
    X = np.array([w.flatten() for w in pssm_submatrix_list])
    y_pred = model.predict(X)  # 0/1

    binding_site_indices = [i for i, pred in enumerate(y_pred) if pred == 1]

    all_protein_results = {}

    all_protein_results[primary_accession] = {
                "atp_binding_site_indices": binding_site_indices,
                "num_predicted_binding_sites": len(binding_site_indices)
            }
    print("\nAll predictions complete.")
    return all_protein_results

def output(all_protein_results):
    return_str = ''
    for protein_id, data in all_protein_results.items():
        print(f"Protein: {protein_id},", end=' ')
        return_str += f"Protein: {protein_id}, "
        if len(data['atp_binding_site_indices']) == 0:
            print("No ATP Binding Site is predicted here.")
            return_str += "No ATP Binding Site is predicted here.\n"
        else:
            # 這裡將每個 index 都加 7
            shifted_indices = [idx + 7 for idx in data['atp_binding_site_indices']]
            print(f"Predicted ATP Binding Sites: {shifted_indices}")
            return_str += f"Predicted ATP Binding Sites: {shifted_indices}\n"
    return return_str