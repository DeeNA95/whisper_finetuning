import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import torch
import whisper
from whisper import DecodingOptions, DecodingResult

from whisper_utils import load_finetuned_model, preprocess_audio, select_device

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging with timestamps and levels."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class LiveTranscriptionConfig:
    """Configuration for live transcription parameters."""

    def __init__(
        self,
        model_size: str = "tiny",
        checkpoint_path: Optional[Path] = None,
        sample_rate: int = 16000,
        chunk_seconds: float = 5.0,
        hop_seconds: float = 2.0,
        preferred_device: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.checkpoint_path = checkpoint_path or Path("checkpoints") / "best_model.pt"
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.hop_seconds = hop_seconds
        self.preferred_device = preferred_device

        self.chunk_samples: int = int(self.sample_rate * self.chunk_seconds)
        self.hop_samples: int = int(self.sample_rate * self.hop_seconds)


def load_whisper_model_for_live_transcription(
    config: LiveTranscriptionConfig,
) -> whisper.Whisper:
    """
    Load the finetuned Whisper model for live inference.

    Args:
        config: Configuration with model details.

    Returns:
        The loaded Whisper model ready for inference.
    """
    logger.info("🧠 ***** Starting Whisper model load for live transcription")
    logger.debug(f"📄 {config.model_size = }")
    logger.debug(f"📄 {config.checkpoint_path = }")

    device = select_device(config.preferred_device)
    logger.debug(f"🖥️ {device = }")

    model = load_finetuned_model(
        model_size=config.model_size,
        checkpoint_path=config.checkpoint_path,
        device=device,
    )
    model.eval()

    logger.info("✅ ***** Whisper model loaded successfully.")
    return model


def create_decoding_options() -> DecodingOptions:
    """
    Create default decoding options for live transcription.

    No forced language, transcription task.

    Returns:
        Configured DecodingOptions.
    """
    logger.info("🧩 ***** Building decoding options for live transcription")
    options = DecodingOptions(
        task="transcribe",
        temperature=0.0,
        without_timestamps=True,
    )
    logger.info("✅ ***** Decoding options ready.")
    return options


def process_audio_chunk_for_transcription(
    chunk: np.ndarray,
    sample_rate: int,
    model: whisper.Whisper,
    options: DecodingOptions,
    max_duration_seconds: float,
) -> Optional[str]:
    """
    Preprocess and transcribe a single audio chunk.

    Args:
        chunk: Audio waveform array.
        sample_rate: Sampling rate of the audio.
        model: Loaded Whisper model.
        options: Decoding parameters.
        max_duration_seconds: Max duration for padding/trimming.

    Returns:
        Transcribed text if non-empty, else None.
    """
    logger.info("🎛️ ***** Processing audio chunk for transcription")
    logger.debug(f"📄 {chunk.shape = }")

    mel = preprocess_audio(
        audio_array=chunk,
        sample_rate=sample_rate,
        target_sample_rate=16000,
        max_duration_seconds=max_duration_seconds,
    ).to(model.device)

    logger.debug(f"🎵 {mel.shape = } on device {mel.device}")

    with torch.no_grad():
        result: DecodingResult = whisper.decode(model, mel, options)

    transcript = result.text.strip()
    if not transcript:
        logger.debug("🕳️ Empty transcript for current chunk.")
        return None

    logger.info(f'✅ Transcript: "{transcript}"')
    return transcript


def live_transcription_loop(
    model: whisper.Whisper,
    config: LiveTranscriptionConfig,
    options: DecodingOptions,
) -> None:
    """
    Main loop for capturing and transcribing live microphone audio.

    Args:
        model: Loaded Whisper model.
        config: Transcription configuration.
        options: Decoding options.
    """
    logger.info("🎙️ ***** Starting live microphone capture and transcription")
    logger.debug(f"📄 {config.chunk_samples = }")
    logger.debug(f"📄 {config.hop_samples = }")

    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=config.sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    audio_buffer = np.zeros(0, dtype=np.float32)
    last_transcript_time = time.time()

    def signal_handler(sig: int, frame: Optional[object]) -> None:
        logger.info("🛑 ***** Stopping live transcription on signal")
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        try:
            raw_data = stream.read(1024, exception_on_overflow=False)
            pcm_samples = (
                np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            audio_buffer = np.concatenate((audio_buffer, pcm_samples))

            while audio_buffer.shape[0] >= config.chunk_samples:
                chunk = audio_buffer[: config.chunk_samples]
                audio_buffer = audio_buffer[config.hop_samples :]

                transcript = process_audio_chunk_for_transcription(
                    chunk=chunk,
                    sample_rate=config.sample_rate,
                    model=model,
                    options=options,
                    max_duration_seconds=30.0,
                )

                if transcript:
                    current_time = time.time()
                    latency_ms = int((current_time - last_transcript_time) * 1000)
                    last_transcript_time = current_time
                    print(f"Transcript (latency {latency_ms}ms): {transcript}")

            if audio_buffer.shape[0] > 0:
                logger.debug(f"🧺 {audio_buffer.shape = } retained for next chunk")

        except Exception as error:
            logger.error(f"❌ Error in live loop: {error}")
            break

    stream.stop_stream()
    stream.close()
    audio_interface.terminate()
    logger.info("✅ ***** Live transcription loop ended.")


def main() -> None:
    """Entry point for live inference script."""
    setup_logging()
    logger.info("🚀 ***** Starting live inference script")

    config = LiveTranscriptionConfig()
    model = load_whisper_model_for_live_transcription(config)
    options = create_decoding_options()

    live_transcription_loop(model, config, options)


if __name__ == "__main__":
    main()
