import ollama
import torch


def get_embeddings(audio_text: str) -> torch.Tensor:
    return ollama.embeddings(model="llama3.1:latest", prompt=audio_text)
