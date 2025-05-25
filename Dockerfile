# ------------------------------------------------------------------------------
# 本 Docker 建構過程會自動 clone 下列第三方程式碼：
#   來源：https://github.com/amazon-science/LC-PLM?tab=License-1-ov-file
#   授權：CC BY-NC 4.0（創用CC姓名標示-非商業性4.0國際授權條款）
# 
# 請注意，該部分內容僅能用於非商業用途，併同原作者授權條款一併遵守。
# 若需商業用途，請直接洽詢原作者。
# ------------------------------------------------------------------------------

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
RUN git clone https://github.com/w39011651/LC-PLM.git /app/LC-PLM
#預設從github上clone下來
#The repository which forked from LC-PLM, it would have n_layer=8

RUN pip install gdown

RUN gdown --id 1Jc5YRHPSnbrSf33-A-LeokZMptShCU89 -O app/mamba_atp_binding_model.pt

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install --no-build-isolation mamba-ssm

COPY ./app /app


EXPOSE 5000
#CMD ["ls", "predict"]

RUN ls -al /app/LC-PLM
#Ensure that LC-PLM is under /app/

CMD ["python", "website_demo.py"]