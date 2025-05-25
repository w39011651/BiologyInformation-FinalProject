import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from cnn_with_bm.preprocess import process_single_protein

def predict(model, protein_info_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_protein_results = {}

    SEQ_LEN = 15
    BEST_THRESHOLD = np.float32(0.742)#The best threshold for the highest MCC score in test phase when the model was training

    model.eval()
    with torch.no_grad():
        for primary_accession, sequence in tqdm(protein_info_list):
            protein_window_tensor, bm_matrix = process_single_protein(sequence, 15)
            #protein_window_tensor: torch.tensor() for every window in the matrix

            protein_window_tensor = protein_window_tensor.unsqueeze(1).to(device)

            window_logits = model(protein_window_tensor)

            window_probs = F.softmax(window_logits, dim=1)[:,1].cpu().numpy()
            #The probability of "is ATP binding site"

            sequence_length = bm_matrix.shape[0]
            residue_probs_sum = np.zeros(sequence_length, dtype=np.float32)#儲存每個殘基的機率總和
            residue_coverage_count = np.zeros(sequence_length, dtype=np.int32)#儲存每個殘基被覆蓋的次數

            center_offset = SEQ_LEN//2

            for i, prob in enumerate(window_probs):
                center_residue_idx = i + center_offset #0 for the first window, 1 for the second window
                residue_probs_sum[center_residue_idx] += prob
                residue_coverage_count[center_residue_idx] += 1
            
            #Calculate the average probability of every residue
            residue_average_probs = np.where(
                residue_coverage_count > 0,
                residue_probs_sum / residue_coverage_count,
                0.0
            )

            residue_prediction = (residue_average_probs > BEST_THRESHOLD)

            atp_binding_site_indices = np.where(residue_prediction == 1)[0].tolist()

            all_protein_results[primary_accession] = {
                "sequence_length": sequence_length,
                "residue_avg_probs": residue_average_probs.tolist(),
                "residue_predictions": residue_prediction.tolist(),
                "atp_binding_site_indices": atp_binding_site_indices,
                "num_predicted_binding_sites": len(atp_binding_site_indices)
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
            print("Predicted ATP Binding Sites: {data['atp_binding_site_indices']}")
            return_str += f"Predicted ATP Binding Sites: {data['atp_binding_site_indices']}\n"
    return return_str