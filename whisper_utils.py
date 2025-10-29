from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import whisper
from whisper.tokenizer import Tokenizer
from datasets import Dataset

logger = logging.getLogger(__name__)

__all__ = [
    "select_device",
    "load_finetuned_model",
    "build_tokenizer",
    "resample_audio",
    "pad_or_trim_audio",
    "preprocess_audio",
    "prepare_audio_file",
    "mel_spectrogram_from_file",
    "chunk_audio_stream",
]


def select_device(preferred_device: Optional[str] = None) -> torch.device:
    """Select the best available torch device.

    Args:
        preferred_device: Optional device string (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``).

    Returns:
        torch.device: Resolved device ready for model placement.

    Raises:
        ValueError: If a preferred device is provided but unavailable.
    """
    logger.info("🚀 ***** Starting Whisper device selection")
    logger.debug(f"📄 {preferred_device = }")

    if preferred_device:
        normalized = preferred_device.lower()
        if normalized.startswith("cuda") and torch.cuda.is_available():
            device = torch.device("cuda")
        elif normalized == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif normalized == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError(
                f'Preferred device "{preferred_device}" requested but not available.'
            )
        logger.debug(f"🖥️ {device = }")
        logger.info("✅ ***** Whisper device selection done.")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.debug(f"🖥️ {device = }")
    logger.info("✅ ***** Whisper device selection done.")
    return device


def load_finetuned_model(
    model_size: str = "tiny",
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> whisper.model.Whisper:
    """Load a Whisper model and apply an optional finetuned checkpoint.

    Args:
        model_size: Whisper checkpoint name (e.g. ``"tiny"``, ``"base"``, ``"small"``).
        checkpoint_path: Optional path to a saved finetuned state dict.
        device: Optional device override. When ``None`` it is auto-selected.

    Returns:
        whisper.model.Whisper: The ready-to-infer Whisper model.
    """
    logger.info("🧠 ***** Starting Whisper model load")
    logger.debug(f"📄 {model_size = }")
    logger.debug(f"📄 {checkpoint_path = }")

    # resolved_device = device or select_device()
    resolved_device = torch.device("cpu")
    logger.debug(f"🖥️ {resolved_device = }")

    model = whisper.load_model(model_size)
    model = model.to(resolved_device)

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=resolved_device)
            model_state = state.get("model_state_dict", state)
            model.load_state_dict(model_state)
            logger.info('📦 Loaded finetuned weights from %s', checkpoint_path)
        else:
            logger.warning(
                "⚠️ Checkpoint %s not found; continuing with base Whisper weights.",
                checkpoint_path,
            )

    logger.info("✅ ***** Whisper model load done.")
    return model


def build_tokenizer(
    model: whisper.model.Whisper,
    language: str = "en",
    task: str = "transcribe",
) -> Tokenizer:
    """Instantiate the Whisper tokenizer aligned with the model configuration.

    Args:
        model: Loaded Whisper model instance.
        language: Target language ISO code (defaults to English).
        task: Whisper decoding task (``"transcribe"`` or ``"translate"``).

    Returns:
        Tokenizer: Configured tokenizer compatible with the model.
    """
    logger.info("🗣️ ***** Starting tokenizer initialization")
    logger.debug(f"📄 {language = }")
    logger.debug(f"📄 {task = }")

    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=model.is_multilingual,
        language=language,
        task=task,
    )

    logger.info("✅ ***** Tokenizer initialization done.")
    return tokenizer


def resample_audio(
    audio_array: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """Resample an audio waveform to the target sample rate.

    Args:
        audio_array: Input waveform.
        original_sample_rate: Original sampling rate of the waveform.
        target_sample_rate: Desired sampling rate for Whisper (default 16 kHz).

    Returns:
        np.ndarray: Resampled mono waveform in float32 precision.
    """
    logger.info("🎧 ***** Starting audio resample")
    logger.debug(f"📄 {original_sample_rate = }")
    logger.debug(f"📄 {target_sample_rate = }")

    if original_sample_rate == target_sample_rate:
        logger.info("✅ ***** Audio resample done (resample skipped).")
        return audio_array.astype(np.float32)

    try:
        import librosa
    except ImportError as error:
        raise ImportError(
            "librosa is required for resampling. Install it via pip install librosa."
        ) from error

    resampled = librosa.resample(
        audio_array.astype(np.float32),
        orig_sr=original_sample_rate,
        target_sr=target_sample_rate,
    )

    logger.info("✅ ***** Audio resample done.")
    return resampled.astype(np.float32)


def pad_or_trim_audio(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    max_duration_seconds: float = 30.0,
) -> np.ndarray:
    """Pad or trim audio to the maximum duration expected by Whisper.

    Args:
        audio_array: Waveform to process.
        sample_rate: Sampling rate of the waveform (defaults to 16 kHz).
        max_duration_seconds: Maximum duration to retain (defaults to 30 s).

    Returns:
        np.ndarray: Waveform padded or trimmed to the target duration.
    """
    logger.info("🪄 ***** Starting pad/trim audio")
    logger.debug(f"📄 {sample_rate = }")
    logger.debug(f"📄 {max_duration_seconds = }")

    target_samples = int(max_duration_seconds * sample_rate)
    if audio_array.shape[0] == target_samples:
        padded = audio_array
    else:
        padded = whisper.pad_or_trim(audio_array)

    logger.info("✅ ***** Pad/trim audio done.")
    return padded.astype(np.float32)


def preprocess_audio(
    audio_array: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = 16000,
    max_duration_seconds: float = 30.0,
) -> torch.Tensor:
    """Complete preprocessing pipeline that resamples, pads, and creates mel features.

    Args:
        audio_array: Raw audio waveform.
        sample_rate: Sampling rate of the input waveform.
        target_sample_rate: Desired sampling rate (defaults to 16 kHz).
        max_duration_seconds: Maximum audio duration considered.

    Returns:
        torch.Tensor: Mel spectrogram tensor with shape ``(80, n_frames)``. 128 for large
    """
    logger.info("🛠️ ***** Starting Whisper audio preprocessing")
    logger.debug(f"📄 {sample_rate = }")
    logger.debug(f"📄 {target_sample_rate = }")

    waveform = resample_audio(
        audio_array=audio_array,
        original_sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
    )
    waveform = pad_or_trim_audio(
        audio_array=waveform,
        sample_rate=target_sample_rate,
        max_duration_seconds=max_duration_seconds,
    )
    mel = whisper.log_mel_spectrogram(waveform, n_mels=128)
    

    logger.info("✅ ***** Whisper audio preprocessing done.")
    return mel


def prepare_audio_file(
    audio_path: Path,
    target_sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, convert to mono float32, and resample if required.

    Args:
        audio_path: Path to the audio file.
        target_sample_rate: Desired sample rate for downstream processing.

    Returns:
        Tuple[np.ndarray, int]: Tuple containing the processed waveform and its sampling rate.
    """
    logger.info("📂 ***** Starting audio file load")
    logger.debug(f"📄 {audio_path = }")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    fallback_waveform: Optional[np.ndarray] = None
    fallback_rate: Optional[int] = None

    try:
        import soundfile as sf  # type: ignore
    except ImportError as error:
        logger.warning(
            "⚠️ soundfile unavailable (%s); falling back to whisper.load_audio.",
            error,
        )
        fallback_waveform = whisper.load_audio(str(audio_path))
        fallback_rate = 16000
    else:
        try:
            waveform, sample_rate = sf.read(audio_path, always_2d=False)
            logger.debug(f"🧾 {sample_rate = }")
        except Exception as error:  # pylint: disable=broad-except
            logger.warning(
                "⚠️ soundfile failed to open %s (%s); falling back to whisper.load_audio.",
                audio_path,
                error,
            )
            fallback_waveform = whisper.load_audio(str(audio_path))
            fallback_rate = 16000
        else:
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            waveform = waveform.astype(np.float32)
            logger.info("✅ ***** Audio file load done.")
            waveform = resample_audio(
                audio_array=waveform,
                original_sample_rate=sample_rate,
                target_sample_rate=target_sample_rate,
            )
            return waveform, target_sample_rate

    assert fallback_waveform is not None
    assert fallback_rate is not None

    logger.info("✅ ***** Audio file load done via fallback.")
    if fallback_rate != target_sample_rate:
        fallback_waveform = resample_audio(
            audio_array=fallback_waveform,
            original_sample_rate=fallback_rate,
            target_sample_rate=target_sample_rate,
        )
        return fallback_waveform, target_sample_rate

    return fallback_waveform.astype(np.float32), fallback_rate


def mel_spectrogram_from_file(
    audio_path: Path,
    target_sample_rate: int = 16000,
    max_duration_seconds: float = 30.0,
) -> torch.Tensor:
    """Convenience helper that loads an audio file and returns its mel spectrogram.

    Args:
        audio_path: Path to the audio file.
        target_sample_rate: Desired sample rate for Whisper.
        max_duration_seconds: Maximum duration to consider.

    Returns:
        torch.Tensor: Mel spectrogram compatible with Whisper encoder input.
    """
    logger.info("🎼 ***** Starting mel spectrogram computation from file")
    logger.debug(f"📄 {audio_path = }")

    waveform, sample_rate = prepare_audio_file(
        audio_path=audio_path, target_sample_rate=target_sample_rate
    )
    mel = preprocess_audio(
        audio_array=waveform,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        max_duration_seconds=max_duration_seconds,
    )

    logger.info("✅ ***** Mel spectrogram computation from file done.")
    return mel


def chunk_audio_stream(
    audio_array: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: Optional[float] = None,
) -> Iterator[np.ndarray]:
    """Yield overlapping audio chunks suitable for streaming transcription.

    Args:
        audio_array: Input waveform.
        sample_rate: Sampling rate of the waveform.
        window_seconds: Size of each chunk window in seconds.
        hop_seconds: Hop size between consecutive windows. Defaults to half the window.

    Yields:
        np.ndarray: Windowed waveform segments.
    """
    logger.info("🪟 ***** Starting audio chunking")
    logger.debug(f"📄 {window_seconds = }")
    logger.debug(f"📄 {hop_seconds = }")

    effective_hop = hop_seconds or window_seconds / 2.0
    window_samples = max(1, int(window_seconds * sample_rate))
    hop_samples = max(1, int(effective_hop * sample_rate))

    total_samples = audio_array.shape[0]
    logger.debug(f"📊 {total_samples = }")

    idx = 0
    while idx < total_samples:
        end = idx + window_samples
        chunk = audio_array[idx:end]
        if chunk.shape[0] < window_samples:
            chunk = np.pad(chunk, (0, window_samples - chunk.shape[0]), mode="constant")
        yield chunk.astype(np.float32)
        idx += hop_samples
        if end >= total_samples:
            break

    logger.info("✅ ***** Audio chunking done.")

class WhisperDataset(Dataset):
    """
    Custom Dataset for Whisper finetuning.

    Handles audio preprocessing and text tokenization.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        sample_rate: int = 16000,
        max_length: int = 448,  # Whisper's typical max length
    ):
        """
        Args:
            dataset: HuggingFace dataset with 'audio' and 'text' features
            tokenizer: Whisper tokenizer
            sample_rate: Target sampling rate (16000 for Whisper)
            max_length: Maximum token sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - 'input_features': mel spectrogram (80, 3000)
                - 'labels': tokenized text (max_length,)
                - 'dec_input_ids': decoder input (shifted labels)
        """
        # Get audio and text
        item = self.dataset[idx]
        audio_array = item["audio"]["array"]
        audio_sr = item["audio"]["sampling_rate"]
        text = item["text"]

        # === AUDIO PREPROCESSING ===
        # Resample if needed (datasets library can do this automatically)
        if audio_sr != self.sample_rate:
            import librosa

            audio_array = librosa.resample(
                audio_array, orig_sr=audio_sr, target_sr=self.sample_rate
            )

        # Convert to mel spectrogram using Whisper's preprocessing
        audio_array = whisper.pad_or_trim(audio_array)
        mel = preprocess_audio(
            audio_array=audio_array,
            sample_rate=audio_sr,
            target_sample_rate=self.sample_rate,
        )

        mel_tensor = (
            mel.clone().detach()
            if isinstance(mel, torch.Tensor)
            else torch.from_numpy(mel).float()
        )

        # === TEXT TOKENIZATION ===
        # Proper teacher forcing setup:
        # Decoder input: SOT + text_tokens
        # Labels: text_tokens + EOT (what model should predict)

        # Encode text
        text_tokens = self.tokenizer.encode(text)

        # Create decoder input (what the model sees)
        dec_input_ids = [self.tokenizer.sot] + text_tokens

        # Create labels (what model should predict, shifted left by 1)
        # When decoder sees SOT, it should predict first token
        # When decoder sees first token, it should predict second token, etc.
        labels = text_tokens + [self.tokenizer.eot]

        # Ensure labels and decoder inputs have same length
        max_seq_len = min(self.max_length, len(dec_input_ids))

        if len(dec_input_ids) > max_seq_len:
            dec_input_ids = dec_input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
        else:
            # Pad decoder input with EOT tokens
            dec_input_ids = dec_input_ids + [self.tokenizer.eot] * (
                max_seq_len - len(dec_input_ids)
            )
            # Pad labels with -100 (ignored in loss)
            labels = labels + [-100] * (max_seq_len - len(labels))

        # Pad to max_length if needed
        if len(dec_input_ids) < self.max_length:
            dec_input_ids = dec_input_ids + [self.tokenizer.eot] * (
                self.max_length - len(dec_input_ids)
            )
            labels = labels + [-100] * (self.max_length - len(labels))

        return {
            "input_features": mel_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "dec_input_ids": torch.tensor(dec_input_ids, dtype=torch.long),
        }
