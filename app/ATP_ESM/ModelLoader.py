import torch
from ATP_ESM import ESMModel

def load_model(model_weight_path = "esm_atp_binding_model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMModel.EsmForSequenceLabeling("facebook/esm2_t6_8M_UR50D", 2)
    model.to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print("Successfully loaded model weight.")
    return model