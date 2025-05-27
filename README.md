## 使用說明

### 本專案未包含 LC-PLM 預訓練模型與原始碼，Docker build 時將自動從 [amazon-science/LC-PLM](https://github.com/amazon-science/LC-PLM) 下載最新版本及權重。如需手動執行，請預先下載並放置於 `/LC-PLM` 目錄。

### 使用:

首先使用`git clone https://github.com/w39011651/BiologyInformation-FinalProject.git`複製repository到想要運行的機器上

接著進入專案資料夾，如: `cd BiologyInformation-FinalProject`

建立Docker映像檔: `docker build -t your-images-name .

執行容器: `docker run --rm --gpus all -p 5000:5000 your-images-name:tag`

如果想要容器在結束後自動重啟: `docker run --restart always --gpus all -p 5000:5000 your-images-name:tag`

### 環境與需求:

+ OS: Linux
+ CUDA 12.1+
+ 需要安裝NVIDIA GPU Driver

## 致謝與授權

本專案使用 [amazon-science/LC-PLM](https://github.com/amazon-science/LC-PLM) 之模型權重與程式碼，  
其內容依 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 授權條款提供。  
詳細授權內容請參閱 LC-PLM 專案的 LICENSE 檔案。

## Instructions

### This project does not include the LC-PLM pre-trained model and source code. The latest version and weights will be automatically downloaded from [amazon-science/LC-PLM](https://github.com/amazon-science/LC-PLM) during Docker build. If you want to execute it manually, please download it in advance and put it in the `/LC-PLM` directory.

### use:

First, use `git clone https://github.com/w39011651/BiologyInformation-FinalProject.git` to copy the repository to the machine you want to run

Then enter the project folder, such as: `cd BiologyInformation-FinalProject`

Build Docker images: `docker build -t your-images-name .

Execute the container: `docker run --rm --gpus all -p 5000:5000 your-images-name:tag`

If you want the container to automatically restart after it ends: `docker run --restart always --gpus all -p 5000:5000 your-images-name:tag`

### Environment and requirements:

+ OS: Linux
+ CUDA 12.1+
+ NVIDIA GPU Driver needs to be installed

## Acknowledgements and Authorization

This project uses the model weights and code from [amazon-science/LC-PLM](https://github.com/amazon-science/LC-PLM).  
Its content is provided under the terms of the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.     
For detailed authorization information, please refer to the LICENSE file of the LC-PLM project.
