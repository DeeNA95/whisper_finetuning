from datasets import Dataset, Features, Audio, Value
from  pathlib import Path
from whisper_utils import WhisperDataset, build_tokenizer
import pandas as pd
import whisper

df = pd.read_csv('fisd-akuapim-twi-90p/data.csv', sep='\t')
# df.head()
for index, row in df[['Audio Filepath', 'Unnamed: 0']].iterrows():
    breaks = row['Audio Filepath'].split('/')
    df.at[index, 'filepath'] = breaks[-1]

# df.head()
model = whisper.load_model("large")
audio_root = Path('fisd-akuapim-twi-90p/audios')

features = Features({
    'audio': Audio(sampling_rate=16000),
    'text': Value('string'),
})

hf_dataset = Dataset.from_dict(
    {
        'audio': [audio_root / fp for fp in df['filepath']],
        'text': df['Transcription'].tolist(),
    },
    features=features,
).cast_column('audio', Audio(sampling_rate=16000))

tokenizer = build_tokenizer(model)

akuapim = WhisperDataset(
    dataset=hf_dataset,
    tokenizer=tokenizer,
    sample_rate=16000,
    max_length=448,
)
first_sample = akuapim[0]

print(first_sample)

save_dir = Path('datasets/akuapim_whisper')
hf_dataset.save_to_disk(save_dir)


#load from locally saved
# loaded_dataset = Dataset.load_from_disk(save_dir)
