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

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install gdown

COPY ./app /app


EXPOSE 5000
#CMD ["ls", "predict"]
CMD ["python", "website_demo.py"]