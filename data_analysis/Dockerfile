FROM python:3.9-slim



RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \ 
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /data_analysis

RUN git clone https://github.com/shravanganji/bdd100k-assignment.git /data_analysis/app

WORKDIR /data_analyis/app

COPY requirments.txt .

RUN pip3 install -r requirments.txt
RUN git clone https://github.com/streamlit/streamlit-example.git app/streamlit
RUN pip3 install gdown


COPY . .

RUN python data_downloader.py
RUN unzip csv_data.zip
RUN rm csv_data.zip


COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
