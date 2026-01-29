FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip wget

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Ladda ner engine-filen från din server
RUN mkdir -p /models && \
    wget https://intra.sonerna.se/wp-content/uploads/beverageValidator/models/best50.engine \
    -O /models/best50.engine

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
