# bdd100k-assignment

Dockers images
1. for data analysis :
   gdrive link : 
   1. docker load -i analysis
   2. docker run -p 8501:8501 analysis

3. For inference
   gdrive link :
   1. docker load -i inference
   2. docker run --gpus all -p 8502:8502 inference
  
If you want to build docker images from scratch for :
1. Data analysis :
   build : docker build -t analysis
   run : docker run -p 8501:8501 analysis
2. Inference :
   build : docker build -t inference
   run : docker run --gpus all inference
   Note : Make sure to install docker-nvidia  
