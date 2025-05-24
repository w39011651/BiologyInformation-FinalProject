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

RUN git clone https://github.com/amazon-science/LC-PLM.git && mv LC-PLM app/
#預設從github上clone下來

RUN pip install gdown

RUN gdown --id 1Jc5YRHPSnbrSf33-A-LeokZMptShCU89 -O app/atp_binding_model.pt

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install --no-build-isolation mamba-ssm

COPY ./app /app


EXPOSE 5000
#CMD ["ls", "predict"]
CMD ["python", "website_demo.py"]