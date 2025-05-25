from transformers import AutoTokenizer

from ATP_Mamba import Dataprocessor
from ATP_Mamba import ModelLoader
from ATP_Mamba import MambaPredictor

def run(protein_information):
    """
    args:
    protein_information: Fasta format like (>sp|ID|protein information...\\n protein sequence)
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")#可以考慮在部屬服務時，load所有模型的權重以及tokenizer

    try:
        primary_accession, sequence = Dataprocessor.preprocess_information(protein_information=protein_information)
    except Exception:
        print("The input format is not valid.")
        return
    
    token = Dataprocessor.ATPtokenize(tokenizer, sequence)
    model = ModelLoader.load_model()
    result = MambaPredictor.predict(model, token)
    output = MambaPredictor.output(primary_accession, result)

    return output
