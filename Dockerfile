# Use an official Python runtime as a parent image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3 python3-pip git make curl ffmpeg
RUN apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git/

RUN mv whisper.cpp ../
RUN cd ../whisper.cpp; make
RUN mkdir /models
RUN cd ../whisper.cpp; ./models/download-ggml-model.sh tiny.en /models

# Copy the rest of the application code to the container
COPY . /app

# Set environment variables for FastAPI
ENV HOST 0.0.0.0
ENV PORT 8000

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

