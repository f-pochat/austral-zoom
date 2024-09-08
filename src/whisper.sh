WHISPER_CPP_PATH="../whisper.cpp/"
TMP_AUDIO_FILE="/app/src/tmp/audio.wav.txt"
TMP_OUTPUT_TXT="/app/src/tmp/audio.wav"
WHISPER_MODEL="tiny.en"

cd "$WHISPER_CPP_PATH" || exit
./main -m /models/ggml-$WHISPER_MODEL.bin -f "$TMP_AUDIO_FILE" -otxt "$TMP_OUTPUT_TXT"