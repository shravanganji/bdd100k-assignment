# Use the official CUDA 11.3.1 base image
FROM nvidia/cuda:11.3.1-base

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y curl

RUN ln -sf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime

RUN apt install -y git 

RUN apt-get install -y python3 python3-pip libgl1-mesa-glx libglib2.0-0 libglib2.0-dev unzip


# Set up the NVIDIA CUDA repository
RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility



# Install Python 3 and pip
#RUN apt-get install -y python3 python3-pip libgl1-mesa-glx libglib2.0-0 libglib2.0-dev tzdata

# Set the working directory
WORKDIR /inference

RUN git clone https://github.com/shravanganji/bdd100k-assignment.git /inference/app

WORKDIR /inference/app

# Copy the requirements file into the container


RUN pip install gdown

copy . .
RUN python3 data_downloader.py
RUN unzip data/sample.zip
RUN rm data/sample.zip

# Install Python dependencies
RUN pip install -r requirements.txt

RUN pip install mmcv-full==1.6.0 -f  https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
RUN pip install mmdet==2.24.0 mmcls==0.23.2
RUN pip install opencv-python
RUN pip install pydantic==1.10.12
RUN pip install plotly streamlit


# Copy the rest of the application files into the container


#COPY data/ /app/data

# Command to run your application
EXPOSE 8502

# Create a script to run both entrypoints sequentially
RUN echo '#!/bin/bash\npython3 test.py configs/det/faster_rcnn_r101_fpn_3x_det_bdd100k.py --format-only --format-dir /inference/app/data/bdd100k/jsons --val-data data/bdd100k/jsons/det_val_cocofmt.json --image_folder data/bdd100k/images/100k/val\nstreamlit run app.py --server.port=8502 --server.address=0.0.0.0' > /inference/run.sh
RUN chmod +x /inference/run.sh

# Run the script as the entrypoint
ENTRYPOINT ["/inference/run.sh"]
