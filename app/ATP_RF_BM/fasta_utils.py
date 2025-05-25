def extract_fasta_sequence(fasta_str):
    """
    輸入一個FASTA字串，回傳純胺基酸序列（自動忽略>開頭標頭與換行）
    """
    lines = fasta_str.strip().splitlines()
    seq = ""
    for line in lines:
        if not line.startswith(">"):
            seq += line.strip()
    return seq

def extract_primary_accession(fasta_str):
    """
    從FASTA字串中提取主要的accession號碼（假設第一行為標頭，格式為">"）
    """
    lines = fasta_str.strip().splitlines()
    if lines and lines[0].startswith(">"):
        header = lines[0][1:]  # 去掉開頭的">"
        parts = header.split("|")
        if len(parts) > 1:
            return parts[1]  # 假設accession號碼在第二個部分
    return None  # 如果沒有找到，返回None