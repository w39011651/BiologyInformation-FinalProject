def extract_fasta_sequence(fasta_str):
    """
    輸入一個FASTA字串，回傳純胺基酸序列（自動忽略>開頭標頭與換行）
    """
    lines = fasta_str.strip().splitlines()
    seq = ""
    for line in lines:
        if not line.startswith(">sp"):
            seq += line.strip()
    return seq