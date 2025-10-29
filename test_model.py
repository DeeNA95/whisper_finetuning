import torch
import whisper
from pathlib import Path
from whisper_utils import load_finetuned_model

# Load the fine-tuned model
# def load_finetuned_model(checkpoint_path: str, model_size: str = "large"):
#     model = whisper.load_model(model_size)
#     checkpoint = torch.load(checkpoint_path, map_location="cpu")
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#     return model

# Load model
model = load_finetuned_model( "large","best_model.pt", "cuda")

# Transcribe audio file
audio_path = "audio/sample.mp3"  # Replace with actual path
result = model.transcribe(audio_path, language="en")
print("Transcription:", result["text"])

# import torch
# ckpt = torch.load("best_model.pt", map_location="mps")
# print(ckpt.keys())
