import time
import os
import subprocess

def wait_until_exists(file_path, timeout=5):
    for _ in range(int(timeout * 10)):
        if os.path.exists(file_path):
            return
        time.sleep(0.1)
    raise TimeoutError(f"File {file_path} not found after {timeout} seconds.")

def make_pssm(fasta_file:str):
    wait_until_exists(fasta_file)
    primary_accession = fasta_file.split('.')[0]#Assume the path format is PrimaryAccession.fasta. Can use Regular Expression to match
    subprocess.run(f"psiblast -db ./swissprot/swissprot -query {fasta_file} -out_ascii_pssm {primary_accession}.txt -num_iterations 3 -num_threads 16",
                   shell=True,
                   check=True)
    #We need to wait until PSSM has been made.
    print(f"Make PSSM {primary_accession}.txt Done.")
    subprocess.run(f"rm {fasta_file}", shell=True, check=True)
    print(f"Remove the fasta file {fasta_file} Done")
    return primary_accession