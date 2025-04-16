# Use the official Python image
FROM python:3.11-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy necessary files into the container
COPY requirements.txt ./
COPY run.py ./
COPY src/ ./src
COPY data/ ./data
COPY tests/ ./tests

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm