FROM python:3.11

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

COPY model.h5 .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]