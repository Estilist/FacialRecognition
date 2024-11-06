FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev

# Install Python dependencies
RUN pip install -U pip wheel
RUN pip install -r requirements.txt
RUN pip install dlib

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]