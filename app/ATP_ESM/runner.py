from transformers import AutoTokenizer
import torch
from ATP_ESM import Dataprocessor
from ATP_ESM import ModelLoader
from ATP_ESM import ESMpredictor

def run_esm(protein_information):
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
    result = ESMpredictor.predict(model, token)
    output = ESMpredictor.output(primary_accession, result)
    
    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    return output

