train:
	export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:${DYLD_LIBRARY_PATH:-}" && uv run whisper_finetune.py