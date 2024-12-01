from fastapi import UploadFile

from io import BytesIO

from src.logger import log

import subprocess
import os


# TODO add concurrency on tmp_file (e.g. adding a id suffix)
def process(audio: UploadFile, tmp_audio_file: str = "src/tmp/audio.wav", tmp_file: str = "src/tmp/audio.wav.txt") -> str:
    log.info("Processing audio file")
    save_audio(audio, tmp_audio_file)

    log.info("Running whisper")
    if os.getenv("ENVIRONMENT") == "dev":
        subprocess.run(["sh", "-c", "./src/whisper.dev.sh"])
    else:
        subprocess.run(["sh", "-c", "./src/whisper.sh"])

    with open(tmp_file, 'r') as file:
        contents = file.read()

    log.info("Removing temp files")
    os.remove(tmp_audio_file)
    os.remove(tmp_file)

    return contents


def save_audio(audio: UploadFile, output_file: str = "src/tmp/audio.wav"):
    contents = audio.file.read()
    audio_file = BytesIO(contents)

    # Save to a temporary file
    temp_input_file = "src/tmp/temp_input.wav"
    with open(temp_input_file, "wb") as f:
        f.write(audio_file.getvalue())

    # Convert the file using FFmpeg
    subprocess.run([
        "ffmpeg",
        "-i", temp_input_file,
        "-ar", "16000",
        "-ac", "1",
        output_file
    ], check=True)

    os.remove(temp_input_file)
