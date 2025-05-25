from ATP_RF_PSSM.fasta_utils import make_fasta
from ATP_RF_PSSM.pssm_generator import make_pssm
from ATP_RF_PSSM.model_loader import load_rf_model
from ATP_RF_PSSM.predictor import predict, output


def run_rf_pssm(protein_information):
    fasta_file_path = make_fasta(protein_information)
    primary_accession = make_pssm(fasta_file_path)
    model = load_rf_model()
    results = predict(primary_accession, model)#先暫定只會輸入一個Fasta file

    return output(results)