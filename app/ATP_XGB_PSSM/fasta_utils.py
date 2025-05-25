import os

def make_fasta(protein_information):
    """
    Assume that protein_information is a FASTA string with only one sequence.
    """
    primary_accession = protein_information.split('|')[1]
    file_path = f"{primary_accession}.fasta"

    with open(file_path, 'w') as f:
        f.write(protein_information)
        f.flush()              # 確保寫入 buffer
        os.fsync(f.fileno())   # 將 buffer 同步到磁碟

    return file_path