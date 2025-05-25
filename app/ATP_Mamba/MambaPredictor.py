import torch
import numpy as np

def predict(model, data, threshold = 0.429):
    """
    args:
    model: LC-PLM model (Mamba model pretrained by protein)
    data: A protein sequence after being tokenized. It has the following keys: input_ids, attention_mask
    threshold: The best threshold which has the best MCC score on the test dataset.
    """
    device = "cuda" # Mamba artchitecture must use GPU
    with torch.no_grad():
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)

        _, logits = model(input_ids = input_ids, attention_mask = attention_mask)

        logits_np = np.array(logits.cpu().numpy())

        probs = 1/(1+np.exp(-logits_np[:,:,1]))# sigmoid function
        preds = (probs > threshold).astype(int).flatten()
        print(preds)
        atp_binding_site = np.where(preds == 1)[0].tolist()

        return atp_binding_site
    
def output(protein_id, data):
    """
    args:
    predict_result: The np.array() with 0/1 labels, from predict() method
    """
    return_str = ''

    print(f"Protein: {protein_id},", end=' ')
    if len(data) == 0:
        print("No ATP Binding Site is predicted here.")
        return_str += "No ATP Binding Site is predicted here.\n"
    else:
        print(f"Predicted ATP Binding Sites: {data}")
        return_str += f"Predicted ATP Binding Sites: {data}\n"

    return return_str
