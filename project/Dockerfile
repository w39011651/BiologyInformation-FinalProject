FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    build-essential \
    python3-dev \
    default-jre \
    ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/swissprot && \
    wget -c https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz -O /app/swissprot/swissprot.tar.gz && \
    wget -c https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz.md5 -O /app/swissprot/swissprot.tar.gz.md5 && \
    wget -c https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot-prot-metadata.json -O /app/swissprot/swissprot-prot-metadata.json && \
    tar -xzvf /app/swissprot/swissprot.tar.gz -C /app/swissprot && \
    rm /app/swissprot/swissprot.tar.gz

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install gdown

COPY ./app /app


EXPOSE 5000
#CMD ["ls", "predict"]
CMD ["python", "website_demo.py"]