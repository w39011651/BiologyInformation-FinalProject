import torch
import cnn_with_bm.ATP_Model as ATP_Model

def load_model(model_weight_path = "atp_binding_model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ATP_Model.ATPBindingCNN(fc_hidden_dim=500)    
    model.to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print("Successfully loaded model weight.")
    return model