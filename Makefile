train:
	export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:${DYLD_LIBRARY_PATH:-}" && uv run whisper_finetune.py

download-akuapem:
	curl -o fisd-akuapim-twi-90p.zip https://fisd-dataset.s3.amazonaws.com/fisd-akuapim-twi-90p.zip
	unzip -o fisd-akuapim-twi-90p.zip
	python akuapem.py
	rm fisd-akuapim-twi-90p.zip
