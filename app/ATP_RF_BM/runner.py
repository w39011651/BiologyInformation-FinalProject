from ATP_RF_BM.fasta_utils import extract_fasta_sequence
from ATP_RF_BM.model_loader import load_rf_model
from ATP_RF_BM.predictor import predict_binding_sites


def run(protein_information):
    seq = extract_fasta_sequence(protein_information)
    model = load_rf_model()
    results = predict_binding_sites(seq, model)#先暫定只會輸入一個Fasta file