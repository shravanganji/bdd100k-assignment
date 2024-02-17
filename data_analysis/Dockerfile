FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*






COPY Data_analysis/requirments.txt .


RUN pip3 install -r requirments.txt
RUN git clone https://github.com/streamlit/streamlit-example.git app/streamlit
RUN pip3 install gdown


COPY Data_analysis/test.py .

RUN python test.py


COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Data_analysis/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
