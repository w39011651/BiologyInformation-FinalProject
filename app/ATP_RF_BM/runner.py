from ATP_RF_BM.fasta_utils import extract_fasta_sequence, extract_primary_accession
from ATP_RF_BM.model_loader import load_rf_model
from ATP_RF_BM.predictor import predict, output


def run(protein_information):
    seq = extract_fasta_sequence(protein_information)
    primary_accession = extract_primary_accession(protein_information)
    model = load_rf_model()
    results = predict(seq, primary_accession, model)#先暫定只會輸入一個Fasta file

    return output(results)