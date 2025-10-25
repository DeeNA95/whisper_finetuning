from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from whisper import DecodingOptions, DecodingResult, decode

from whisper_utils import (
    chunk_audio_stream,
    prepare_audio_file,
    preprocess_audio,
    select_device,
    load_finetuned_model,
)

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES: Tuple[str, ...] = (
    '.wav',
    '.mp3',
    '.m4a',
    '.flac',
    '.ogg',
    '.opus',
    '.webm',
)


def configure_logging(verbosity: int) -> None:
    """Configure the root logger using verbosity count.

    Args:
        verbosity: Verbosity flag count from CLI (0=INFO, 1+=DEBUG).
    """
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    )
    logger.debug(f'📄 {verbosity = }')
    logger.info('✅ ***** Logging configuration done.')


def collect_audio_files(
    input_path: Path,
    recursive: bool,
    allowed_suffixes: Sequence[str] = AUDIO_SUFFIXES,
) -> List[Path]:
    """Collect audio files from a path.

    Args:
        input_path: File or directory supplied by the user.
        recursive: Whether to scan directories recursively.
        allowed_suffixes: Allowed filename suffixes.

    Returns:
        List[Path]: Sorted list of audio file paths.

    Raises:
        FileNotFoundError: If the input path does not exist.
    """
    logger.info('🗂️ ***** Starting audio file discovery')
    logger.debug(f'📄 {input_path = }')
    logger.debug(f'📄 {recursive = }')

    if not input_path.exists():
        raise FileNotFoundError(f'Input path not found: {input_path}')

    if input_path.is_file():
        files = [input_path]
    else:
        globber: Iterable[Path]
        globber = input_path.rglob('*') if recursive else input_path.glob('*')
        files = [
            candidate
            for candidate in globber
            if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes
        ]

    files.sort()
    logger.info('✅ ***** Audio file discovery done.')
    logger.debug(f'📄 {files = }')
    return files


def build_decoding_options(
    language: str,
    task: str,
    temperature: float,
    beam_size: int,
    best_of: int,
    without_timestamps: bool,
) -> DecodingOptions:
    """Create Whisper decoding options from CLI settings.

    Args:
        language: Target language ISO code.
        task: Whisper task (transcribe or translate).
        temperature: Sampling temperature.
        beam_size: Beam size for beam search.
        best_of: Number of candidates when sampling.
        without_timestamps: Disable timestamp generation.

    Returns:
        DecodingOptions: Configured decoding options instance.
    """
    logger.info('🧩 ***** Starting decoding options build')

    requested_beam = max(1, beam_size)
    requested_best_of = max(1, best_of)

    use_beam = requested_beam > 1
    use_best_of = requested_best_of > 1

    if use_beam and use_best_of:
        if temperature == 0.0:
            logger.warning(
                '⚠️ Both beam search (beam_size=%s) and best-of sampling (best_of=%s) '
                'requested with deterministic decoding; disabling best-of to honor beam search.',
                requested_beam,
                requested_best_of,
            )
            use_best_of = False
        else:
            logger.warning(
                '⚠️ Both beam search (beam_size=%s) and best-of sampling (best_of=%s) '
                'requested; disabling beam search in favor of stochastic best-of sampling.',
                requested_beam,
                requested_best_of,
            )
            use_beam = False

    if use_best_of and temperature == 0.0:
        logger.warning(
            '⚠️ best_of > 1 requires non-zero temperature; switching to beam search with beam_size=%s.',
            requested_best_of,
        )
        use_best_of = False
        use_beam = True
        requested_beam = max(requested_beam, requested_best_of)

    adjusted_beam_size: Optional[int] = requested_beam if use_beam else None
    adjusted_best_of: Optional[int] = requested_best_of if use_best_of else None

    if not use_beam and adjusted_beam_size is None:
        adjusted_beam_size = 1  # whisper expects an int when beam search disabled

    options = DecodingOptions(
        language=language,
        task=task,
        temperature=temperature,
        beam_size=adjusted_beam_size,
        best_of=adjusted_best_of,
        without_timestamps=without_timestamps,
    )
    logger.debug(f'📄 {options = }')
    logger.info('✅ ***** Decoding options build done.')
    return options


def transcribe_file(
    model,
    audio_path: Path,
    decoding_options: DecodingOptions,
    device: torch.device,
    target_sample_rate: int,
    max_duration_seconds: Optional[float],
    chunk_seconds: float,
    hop_seconds: Optional[float],
) -> str:
    """Transcribe a single audio file.

    Args:
        model: Loaded Whisper model.
        audio_path: Path to the input audio file.
        decoding_options: Whisper decoding options.
        device: Torch device where the model resides.
        target_sample_rate: Sample rate for preprocessing.
        max_duration_seconds: Optional maximum duration to process.
        chunk_seconds: Chunk window size in seconds (clamped to 30s).
        hop_seconds: Optional hop size between chunks.

    Returns:
        str: Concatenated transcript text for the entire audio clip.
    """
    logger.info('🎙️ ***** Starting transcription for %s', audio_path)

    waveform, sample_rate = prepare_audio_file(
        audio_path=audio_path,
        target_sample_rate=target_sample_rate,
    )

    if max_duration_seconds and max_duration_seconds > 0:
        max_samples = int(max_duration_seconds * sample_rate)
        waveform = waveform[:max_samples]

    effective_chunk = max(0.1, min(chunk_seconds, 30.0))
    if chunk_seconds > 30.0:
        logger.warning(
            '⚠️ chunk_seconds %.2fs exceeds Whisper receptive field; clamping to 30.0s.',
            chunk_seconds,
        )

    effective_hop = hop_seconds if hop_seconds and hop_seconds > 0 else effective_chunk

    segments: List[str] = []
    for idx, chunk in enumerate(
        chunk_audio_stream(
            audio_array=waveform,
            sample_rate=sample_rate,
            window_seconds=effective_chunk,
            hop_seconds=effective_hop,
        ),
        start=1,
    ):
        logger.info('🎛️ ***** Decoding chunk %s for %s', idx, audio_path)

        mel = preprocess_audio(
            audio_array=chunk,
            sample_rate=sample_rate,
            target_sample_rate=target_sample_rate,
            max_duration_seconds=effective_chunk,
        ).to(device)

        result = decode(model, mel, decoding_options)
        text = result.text.strip()
        logger.debug('📝 Chunk %s text: %s', idx, text)

        if text:
            segments.append(text)

    transcript = ' '.join(segments).strip()
    logger.info('✅ ***** Transcription done for %s', audio_path)
    return transcript


def write_transcript(
    output_dir: Optional[Path],
    audio_path: Path,
    transcript: str,
) -> Optional[Path]:
    """Write transcript to stdout or a text file.

    Args:
        output_dir: Optional directory where transcripts are saved.
        audio_path: Audio source path.
        transcript: Generated transcript text.

    Returns:
        Optional[Path]: Path to the written file if saved.
    """
    if output_dir is None:
        print(f'--- {audio_path} ---')
        print(transcript.strip())
        print()
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{audio_path.stem}.txt'
    output_path.write_text(transcript.strip() + '\n', encoding='utf-8')
    logger.info('📝 Wrote transcript to %s', output_path)
    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Batch transcription using a finetuned Whisper model.',
    )
    parser.add_argument(
        'input_path',
        type=Path,
        help='Audio file or directory containing audio files.',
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search for audio files in subdirectories.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory where transcripts will be written as .txt files.',
    )
    parser.add_argument(
        '--model-size',
        default='tiny',
        help='Whisper model size to load (default: tiny).',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Optional path to finetuned checkpoint weights.',
    )
    parser.add_argument(
        '--device',
        help='Preferred device (cuda, mps, cpu). Auto-detected when omitted.',
    )
    parser.add_argument(
        '--language',
        default='en',
        help='Target language for decoding (default: en).',
    )
    parser.add_argument(
        '--task',
        default='transcribe',
        choices=['transcribe', 'translate'],
        help='Decoding task (transcribe or translate).',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature for decoding.',
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=5,
        help='Beam size for beam search decoding.',
    )
    parser.add_argument(
        '--best-of',
        type=int,
        default=1,
        help='Number of candidates when sampling (requires temperature > 0).',
    )
    parser.add_argument(
        '--without-timestamps',
        action='store_true',
        help='Disable timestamp generation for outputs.',
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Target sample rate for preprocessing (default: 16000).',
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=None,
        help='Optional maximum clip duration (seconds) to process per file before truncation.',
    )
    parser.add_argument(
        '--chunk-seconds',
        type=float,
        default=30.0,
        help='Chunk size (seconds) for long audio (default: 30, clamped to 30).',
    )
    parser.add_argument(
        '--hop-seconds',
        type=float,
        default=None,
        help='Optional hop size between chunks. Defaults to chunk size when omitted.',
    )
    parser.add_argument(
        '--verbose',
        action='count',
        default=0,
        help='Increase logging verbosity (use -v or -vv).',
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the batch transcription CLI."""
    args = parse_arguments()
    configure_logging(args.verbose)

    logger.info('🚀 ***** Starting batch transcription run')
    audio_files = collect_audio_files(args.input_path, args.recursive)
    if not audio_files:
        raise FileNotFoundError(
            f'No audio files found in {args.input_path} with suffixes {AUDIO_SUFFIXES}',
        )

    device = select_device(args.device)
    logger.debug(f'🖥️ {device = }')

    model = load_finetuned_model(
        model_size=args.model_size,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    model.eval()

    decoding_options = build_decoding_options(
        language=args.language,
        task=args.task,
        temperature=args.temperature,
        beam_size=args.beam_size,
        best_of=args.best_of,
        without_timestamps=args.without_timestamps,
    )

    for audio_file in audio_files:
        transcript = transcribe_file(
            model=model,
            audio_path=audio_file,
            decoding_options=decoding_options,
            device=device,
            target_sample_rate=args.sample_rate,
            max_duration_seconds=args.max_duration,
            chunk_seconds=args.chunk_seconds,
            hop_seconds=args.hop_seconds,
        )
        write_transcript(args.output_dir, audio_file, transcript)

    logger.info('✅ ***** Batch transcription run done.')


if __name__ == '__main__':
    main()
