# bdd100k-assignment

Dockers images

   1. For data analysis :
      1. docker pull shravanganji/analysis:v1.0
      2. docker run -p 8501:8501 shravanganji/analysis:v1.0

   2. For inference
      1. docker pull shravanganji/inference:v1.0
      2. docker run --gpus all -p 8502:8502 shravanganji/inference:v1.0
  
To build docker images from scratch for :
   1. Data analysis(data_analysis folder) :
      1. build : docker build -t analysis
      2. run : docker run -p 8501:8501 analysis
   2. Inference(inference folder) :
      1. build : docker build -t inference
      2. run : docker run --gpus all -p 8502:8502 inference

Folder structure:

bdd100k-assignment:
   1. data_analysis - For data analysis purpose
   2. inference - For running inference on pretrained model
   3. retraining - For retraining detectron2 on dataset using FRCNN
   4. Assignment_Documentation.pdf


   
