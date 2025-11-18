FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ src/

COPY models/ models/

EXPOSE 5000

CMD ["python", "src/predict.py"]
