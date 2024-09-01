from fastapi import UploadFile

from src.logger import log
import os
import torch
from transformers import pipeline
import numpy as np
import soundfile as sf
from io import BytesIO


def process(audio: UploadFile, output_file: str = "audio.txt", dev_mode: bool = True) -> str:
    # Check if the output file already exists
    if dev_mode and os.path.exists(output_file):
        log.info(f"File {output_file} exists. Loading from file.")
        with open(output_file, 'r') as f:
            return f.read()

    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # Initialize the ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        chunk_length_s=30,
        device=torch.device(device),
    )

    contents = audio.file.read()

    # Convert the bytes to a file-like object
    audio_file = BytesIO(contents)

    # Load the audio file into an array
    audio, sr = sf.read(audio_file)

    # Ensure audio is mono (Whisper expects mono audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio_content = audio.astype(np.float32)

    prediction = pipe(audio_content)["text"]

    # Optionally, return timestamps along with the text predictions
    prediction_with_timestamps = pipe(audio_content, return_timestamps=True)["chunks"]

    # Save the prediction to a file
    with open(output_file, 'w') as f:
        f.write(prediction)

    log.info(f"Transcription saved to {output_file}")

    return prediction
