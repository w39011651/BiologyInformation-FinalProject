from cnn_with_bm.fasta_utils import extract_fasta_sequence, extract_primary_accession
from cnn_with_bm.model_loader import load_model
from cnn_with_bm.predictor import predict, output


def run(protein_information):
    seq = extract_fasta_sequence(protein_information)
    primary_accession = extract_primary_accession(protein_information)
    model = load_model()
    results = predict(model, [(primary_accession, seq)])#先暫定只會輸入一個Fasta file

    return output(results)