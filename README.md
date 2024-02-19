# bdd100k-assignment

Dockers images
Note : Make sure to install docker-nvidia

   1. for data analysis :
      docker pull docker pull shravanganji/analysis:v1.0
      docker run -p 8501:8501 analysis:v1.0

   2. For inference
      docker pull shravanganji/inference:v1.0
      docker run –gpus all -p 8502:8502 inference:v1.0
  
If you want to build docker images from scratch for :
   1. Data analysis :
      build : docker build -t analysis
      run : docker run -p 8501:8501 analysis
   2. Inference :
      build : docker build -t inference
      run : docker run --gpus all -p 8502:8502 inference

Folder structure:

bdd100k-assignment:
   data_analysis
   inference
   retraining
   Assignment_Documentation_Shravan.pdf


   
