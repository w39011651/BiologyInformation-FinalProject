from predict.fasta_utils import make_fasta
from predict.pssm_generator import make_pssm
from predict.model_loader import load_model
from predict.predictor import predict, output


def run(protein_information):
    fasta_file_path = make_fasta(protein_information)
    primary_accession = make_pssm(fasta_file_path)
    model = load_model()
    results = predict(model, [primary_accession])#先暫定只會輸入一個Fasta file
    return output(results)