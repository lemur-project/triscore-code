FROM python:3.11.12
# Install system dependencies
RUN apt update && apt install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    poppler-utils \
    && apt clean

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
# instal pytorch with gpu
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# install transformers from source
RUN pip install git+https://github.com/huggingface/transformers
# Install python dependencies
RUN pip install -r requirements.txt