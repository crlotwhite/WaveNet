import librosa
import numpy as np

from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # batch는 [(mel_spec, waveform), (mel_spec, waveform), ...] 형태입니다.
    mel_specs = [item['mel_spectrogram'].transpose(0, 1) for item in batch]
    waveforms = [item['waveform'] for item in batch]

    sizes = [mel.size() for mel in mel_specs]

    # Mel-spectrogram과 waveform의 길이를 맞추기 위해 패딩을 적용합니다.
    mel_specs_padded = pad_sequence(mel_specs, batch_first=True, padding_value=0.0)  # (batch_size, n_mels, max_time)
    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)  # (batch_size, max_time)
    mel_specs_padded = mel_specs_padded.transpose(1, 2)

    return mel_specs_padded, waveforms_padded

def preprocess_audio(example, cfg):
    audio = example['audio']
    waveform = audio['array']
    sr = audio['sampling_rate']

    if sr != cfg.data.sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=cfg.data.sample_rate)

    # Mel-spectrogram 변환
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg.data.sample_rate,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        n_mels=cfg.data.n_mels
    )

    # dB scale로 변환
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 정규화
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    example['mel_spectrogram'] = mel_spec_db
    example['waveform'] = waveform

    return example

def load_data(cfg):
    ds = load_dataset('keithito/lj_speech', split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=cfg.data.sample_rate))
    ds = ds.map(preprocess_audio, fn_kwargs={'cfg': cfg}, remove_columns=['audio', 'file', 'text', 'normalized_text'])
    ds = ds.with_format('torch')
    ds.train_test_split(test_size=0.2, random_state=42)
    train_set, test_set = ds['train'], ds['test']
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=cfg.test.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader

