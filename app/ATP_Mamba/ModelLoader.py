import torch
from ATP_Mamba import MambaModel

def load_model(model_weight_path = './atp_binding_model.pt'):
    device = "cuda" #Mamba must use cuda
    model = MambaModel.LCPLMforSequenceLabeling("LC-PLM", num_labels = 2)
    state_dict = torch.load(model_weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    # Load and create the model instance and the state dict at cpu, then move them to GPU together
    model.to(device)
    print("Successfully load model weight.")
    return model