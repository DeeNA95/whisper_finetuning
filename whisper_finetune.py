import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import whisper
from datasets import Audio, load_dataset
from datasets import Dataset as hfDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import modal

app = modal.App(name="whisper-finetune")

# Add required functions from whisper_utils
def select_device(preferred_device=None):
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def build_tokenizer(model, language=None, task="transcribe"):
    import whisper
    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=model.is_multilingual,
        language=language,
        task=task,
    )
    return tokenizer

def preprocess_audio(audio_array, sample_rate, target_sample_rate=16000):
    import numpy as np
    import whisper
    if sample_rate != target_sample_rate:
        import librosa
        audio_array = librosa.resample(
            audio_array, orig_sr=sample_rate, target_sr=target_sample_rate
        )
    audio_array = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_array, n_mels=128)
    return mel

# Define the image with all dependencies
CACHE_DIR = "/cache"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")  # Required for torchcodec
    .pip_install_from_requirements("requirements.txt")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads from Hugging Face
            "HF_HOME": CACHE_DIR,
        }
    )
)

# Set up volumes for datasets and checkpoints
volumes = {
    "/datasets": modal.Volume.from_name("akuapim-dataset", create_if_missing=True),
    "/checkpoints": modal.Volume.from_name("whisper-checkpoints", create_if_missing=True),
    "/cache": modal.Volume.from_name("hf-cache", create_if_missing=True),
}
# https://modal.com/docs/reference
logger = logging.getLogger(__name__)

@app.function(
    image=image,
    gpu="H100",
    volumes=volumes,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=3 * 60 * 60,  # 3 hours
)
def train():
    device = "cuda"
    print(f"Using device: {device}")

    model = whisper.load_model("large")
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

        # Unfreeze last encoder layer
    for param in model.encoder.blocks[-2:].parameters():
        param.requires_grad = True

    # Unfreeze last 4 decoder layers (not just 1)
    for param in model.decoder.blocks[-4:].parameters():
        param.requires_grad = True

    # Both layer norms
    for param in model.encoder.ln_post.parameters():
        param.requires_grad = True
    for param in model.decoder.ln.parameters():
        param.requires_grad = True
        # Verify
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")


    # load dataset
    # dataset = load_dataset("MrDragonFox/Elise")
    dataset = hfDataset.load_from_disk('/datasets/akuapim_whisper/')
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    tokenizer = build_tokenizer(model, language=None, task="transcribe")


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
        # https://modal.com/docs/examples/fine_tune_asr
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


    def whisper_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function to create batches with proper padding.

        Args:
            batch: List of samples from dataset

        Returns:
            Batched tensors with padding
        """
        # Stack mel spectrograms (all same size: 80 x 3000)
        input_features = torch.stack([item["input_features"] for item in batch])

        # Stack labels (already padded to max_length)
        labels = torch.stack([item["labels"] for item in batch])

        # Stack decoder inputs
        dec_input_ids = torch.stack([item["dec_input_ids"] for item in batch])

        return {
            "input_features": input_features,  # (batch, 80, 3000)
            "labels": labels,  # (batch, max_length)
            "dec_input_ids": dec_input_ids,  # (batch, max_length)
        }


    from datasets import DatasetDict


    # Method 1: Using datasets library split
    def create_train_val_split(dataset, test_size=0.2, seed=42):
        """
        Split dataset into train and validation sets.

        Args:
            dataset: HuggingFace dataset
            test_size: Fraction for validation (0.2 = 20%)
            seed: Random seed for reproducibility

        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
        # If dataset already has splits, use them
        if isinstance(dataset, DatasetDict):
            if "train" in dataset and "test" in dataset:
                return dataset
            dataset = dataset

        # Create split
        split_dataset = dataset.train_test_split(
            test_size=test_size, seed=seed, shuffle=True
        )

        # Rename 'test' to 'validation'
        return DatasetDict(
            {"train": split_dataset["train"], "validation": split_dataset["test"]}
        )


    split_dataset = create_train_val_split(
        dataset,
        test_size=0.2,
        seed=42,  # Assuming single 'train' split
    )

    print(f"📊 Dataset splits:")
    print(f"   Train: {len(split_dataset['train'])} samples")
    print(f"   Val:   {len(split_dataset['validation'])} samples")

    # === 3. CREATE PYTORCH DATASETS ===
    train_dataset = WhisperDataset(
        split_dataset["train"], tokenizer, sample_rate=16000, max_length=448
    )

    val_dataset = WhisperDataset(
        split_dataset["validation"], tokenizer, sample_rate=16000, max_length=448
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Small batch for limited data
        shuffle=True,
        num_workers=2,  # Adjust based on CPU cores
        collate_fn=whisper_collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=whisper_collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    print(f"✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"\n📦 Batch shapes:")
    print(f"   input_features: {batch['input_features'].shape}")  # (batch, 80, 3000)
    print(f"   labels: {batch['labels'].shape}")  # (batch, max_length)
    print(f"   dec_input_ids: {batch['dec_input_ids'].shape}")  # (batch, max_length)

    # # Verify label/decoder input alignment
    # print("\n🔍 Verifying label/decoder input alignment...")
    # for i in range(min(10, batch["labels"].shape[0])):  # Check first 10 samples
    #     dec_input = batch["dec_input_ids"][i]
    #     labels = batch["labels"][i]

    #     print(f"\nSample {i}:")
    #     print(f"   Decoder input first 10: {dec_input[:10].tolist()}")
    #     print(f"   Labels first 10: {labels[:10].tolist()}")

    #     # Check alignment
    #     print(
    #         f"   Decoder starts with SOT ({tokenizer.sot})? {dec_input[0].item() == tokenizer.sot}"
    #     )
    #     print(
    #         f"   Labels start with text token (not SOT)? {labels[0].item() != tokenizer.sot}"
    #     )
    #     print(f"   No -100 at position 0? {labels[0].item() != -100}")

    #     # Check teacher forcing alignment
    #     # dec_input[0] should predict labels[0]
    #     # dec_input[1] should predict labels[1], etc.
    #     aligned = True
    #     for j in range(min(5, len(dec_input) - 1)):
    #         if labels[j].item() == -100:
    #             break
    #         if j < len(dec_input) - 1 and dec_input[j + 1].item() != labels[j].item():
    #             aligned = False
    #             print(
    #                 f"   Mismatch at position {j}: dec_input[{j + 1}]={dec_input[j + 1].item()} != labels[{j}]={labels[j].item()}"
    #             )

    #     if aligned:
    #         print("   ✅ Teacher forcing alignment looks correct!")
    #     else:
    #         print("   ❌ Teacher forcing alignment issue detected!")


    def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            model: Whisper model
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total_tokens = 0
        logger.debug("***** Starting train_epoch *****")

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_features = batch["input_features"].to(device)
            dec_input_ids = batch["dec_input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            optimizer.zero_grad()

            # Whisper forward: audio_features = encoder(mel), logits = decoder(tokens, audio_features)
            logger.debug(f"📏 {input_features.shape = } before encoder")
            logger.debug(f"🧠 Model conv1 weight shape: {model.encoder.conv1.weight.shape}")
            audio_features = model.encoder(input_features)
            logits = model.decoder(dec_input_ids, audio_features)

            # Compute loss (ignore padding tokens with label=-100)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()

            # Calculate accuracy (excluding padding)
            mask = labels != -100
            predictions = logits.argmax(dim=-1)
            correct += ((predictions == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | Time: {elapsed:.2f}s"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total_tokens if total_tokens > 0 else 0.0

        return avg_loss, accuracy


    def evaluate(
        model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str
    ) -> Tuple[float, float]:
        """
        Evaluate the model on validation set.

        Args:
            model: Whisper model
            val_loader: Validation data loader
            criterion: Loss function
            device: Device to evaluate on

        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total_tokens = 0
        logger.debug("***** Starting evaluate *****")

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_features = batch["input_features"].to(device)
                dec_input_ids = batch["dec_input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                logger.debug(f"📏 {input_features.shape = } before encoder in evaluate")
                logger.debug(
                    f"🧠 Model conv1 weight shape in evaluate: {model.encoder.conv1.weight.shape}"
                )
                audio_features = model.encoder(input_features)
                logits = model.decoder(dec_input_ids, audio_features)

                # Compute loss
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                total_loss += loss.item()

                # Calculate accuracy
                mask = labels != -100
                predictions = logits.argmax(dim=-1)
                correct += ((predictions == labels) & mask).sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total_tokens if total_tokens > 0 else 0.0

        return avg_loss, accuracy


    # Initialize optimizer and loss function
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.01
    )

    # Cross-entropy loss (ignores -100 labels)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Learning rate scheduler
    num_epochs = 3
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("\n" + "=" * 60)
    print("🚀 TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Batch size: 4")
    print(f"Learning rate: 1e-5")
    print(f"Optimizer: AdamW (weight_decay=0.01)")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Loss function: CrossEntropyLoss")
    print(f"Number of epochs: {num_epochs}")
    print("=" * 60)

    # Test single training step
    # print("\n🧪 Running test training step...")
    # try:
    #     test_batch = next(iter(train_loader))
    #     input_features = test_batch["input_features"].to(device)
    #     dec_input_ids = test_batch["dec_input_ids"].to(device)
    #     labels = test_batch["labels"].to(device)

    #     model.train()
    #     optimizer.zero_grad()

    #     audio_features = model.encoder(input_features)
    #     logits = model.decoder(dec_input_ids, audio_features)
    #     loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

    #     loss.backward()
    #     optimizer.step()

    #     print(f"✅ Test step completed successfully!")
    #     print(f"   Loss: {loss.item():.4f}")
    #     print(f"   Logits shape: {logits.shape}")
    #     print(f"   Labels shape: {labels.shape}")
    # except Exception as e:
    #     print(f"❌ Test step failed: {e}")
    #     raise

    print("\n" + "=" * 60)
    print("✅ Setup complete! Ready to train.")
    print("=" * 60)

    # === FULL TRAINING LOOP ===
    print("\n" + "=" * 60)
    print("🎯 STARTING TRAINING")
    print("=" * 60)

    best_val_loss = float("inf")
    logger.debug("***** After model load *****")
    logger.debug(
        f"🧠 Model dims: n_mels={model.dims.n_mels}, n_audio_ctx={model.dims.n_audio_ctx}"
    )
    logger.debug(
        f"🧠 Encoder conv1 in_channels={model.encoder.conv1.in_channels}, out_channels={model.encoder.conv1.out_channels}"
    )
    checkpoint_dir = Path("/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc * 100:.2f}%")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model_run2.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            print(f"   💾 Checkpoint saved to {checkpoint_path}")

        # Save latest checkpoint
        latest_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_run2.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            },
            latest_checkpoint_path,
        )

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"   dec_input_ids: {batch['dec_input_ids'].shape}")  # (batch, max_length)

    # Commit volumes to ensure checkpoints are saved
    volumes["/checkpoints"].commit()


@app.local_entrypoint()
def main():
    """Run Whisper fine-tuning on Modal."""
    train.remote()
