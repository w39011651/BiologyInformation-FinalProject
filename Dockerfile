FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    build-essential \
    python3-dev \
    default-jre \
    ncbi-blast+ \
    git-lfs \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/swissprot && \
    wget -c https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz -O /app/swissprot/swissprot.tar.gz && \
    wget -c https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz.md5 -O /app/swissprot/swissprot.tar.gz.md5 && \
    wget -c https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot-prot-metadata.json -O /app/swissprot/swissprot-prot-metadata.json && \
    tar -xzvf /app/swissprot/swissprot.tar.gz -C /app/swissprot && \
    rm /app/swissprot/swissprot.tar.gz
RUN pip install weblogo && apt update && apt install -y ghostscript

#RUN git clone https://github.com/amazon-science/LC-PLM.git && mv LC-PLM app/ You can choose origin repository, it would have n_layer=48
RUN git clone https://github.com/w39011651/LC-PLM.git /app/LC-PLM
#預設從github上clone下來
#The repository which forked from LC-PLM, it would have n_layer=8

RUN pip install gdown

RUN gdown --id 1Jc5YRHPSnbrSf33-A-LeokZMptShCU89 -O /app/mamba_atp_binding_model.pt

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install --no-build-isolation mamba-ssm

COPY ./app /app

RUN ls -al /app/LC-PLM
#Ensure that LC-PLM is under /app/

EXPOSE 5000
RUN python -c "import torch; print(torch.cuda.is_available())"
#CMD ["ls", "predict"]
CMD ["python", "website_demo.py"]