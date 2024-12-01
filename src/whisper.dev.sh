WHISPER_CPP_PATH="$HOME/faculty/thesis/whisper.cpp/"
TMP_AUDIO_FILE="$HOME/faculty/thesis/processor/src/tmp/audio.txt"
TMP_OUTPUT_TXT="$HOME/faculty/thesis/processor/src/tmp/audio.wav"
WHISPER_MODEL="large-v2"

cd "$WHISPER_CPP_PATH" || exit
./main -m models/ggml-$WHISPER_MODEL.bin -f "$TMP_AUDIO_FILE" -otxt "$TMP_OUTPUT_TXT"  -l es