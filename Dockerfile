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

#RUN git clone https://github.com/amazon-science/LC-PLM.git && mv LC-PLM app/ You can choose origin repository, it would have n_layer=48
RUN git clone https://github.com/w39011651/LC-PLM.git && mv LC-PLM app/
#預設從github上clone下來
#The repository which forked from LC-PLM, it would have n_layer=8

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