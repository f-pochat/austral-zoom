import ollama
import torch

from src.logger import log


def get_embeddings(audio_text: str) -> torch.Tensor:
    log.info("Generating embeddings for audio text")
    return ollama.embeddings(model="llama3.1:latest", prompt=audio_text)
