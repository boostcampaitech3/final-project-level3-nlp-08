FROM python:3.8.7-slim-buster
COPY . ./app
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get -y install build-essential && \
    pip install --upgrade pip && \
    pip install --default-timeout=1000 -r ./requirements.txt

EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app"]
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8000", "--server.maxUploadSize", "1028"]
