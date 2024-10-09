import librosa
import math
import torch
import torch.nn.functional as F

from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def mulaw(audio):
    # 오디오 신호를 8비트 μ-law 양자화
    # https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    mu = 255
    quantized = torch.floor((torch.sign(audio) * torch.log(1 + mu * torch.abs(audio)) / math.log(1 + mu) + 1) / 2 * 256)
    return quantized.long()

def collate_fn(batch):
    """
    WaveNet용 collate_fn 함수.
    각 샘플의 길이를 맞추기 위해 시퀀스에 패딩을 추가하고, μ-law 전처리된 데이터를
    원-핫 인코딩하여 배치 형태로 변환합니다.

    Parameters:
    - batch: 데이터셋에서 샘플링된 (오디오 시퀀스, 길이) 쌍의 리스트

    Returns:
    - inputs: 패딩된 오디오 데이터 (B, C, T)
    - targets: 시퀀스의 타겟 값 (B, T)
    - lengths: 원본 시퀀스의 길이
    """

    # 각 샘플에서 quantized와 length를 분리
    sequences = [torch.tensor(example['quantized']) for example in batch]
    lengths = [example['length'] for example in batch]

    # 오디오 데이터를 패딩하고 동일한 길이로 맞추기
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    # 패딩된 길이에 맞게 시퀀스를 변환
    # 각 시퀀스를 원-핫 인코딩하여 채널 차원 추가 (B, C, T) 형태
    one_hot_inputs = F.one_hot(padded_sequences, num_classes=256).float().permute(0, 2, 1)

    # 타겟은 입력의 오른쪽으로 한 칸 이동 (미래 값을 예측)
    targets = padded_sequences[:, 1:]
    inputs = one_hot_inputs[:, :, :-1]  # 마지막 타임스텝을 제외한 입력 시퀀스

    # 길이를 텐서로 변환하여 반환
    lengths = torch.tensor(lengths)

    return inputs, targets, lengths

def preprocess_audio(example, cfg):
    """
    학습을 위해 사전 mu-law 양자화 과정 수행

    Parameters:
    - example: 한 행에 해당하는 데이터
    - cfg: 데이터 전처리 관련 하이퍼파라미터 config

    Returns:
    - example: 수정된 example 반환
    """
    audio = example['audio']
    waveform = audio['array']
    sr = audio['sampling_rate']

    if sr != cfg.data.sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=cfg.data.sample_rate)

    quantized = mulaw(waveform)

    example['quantized'] = quantized
    example['length'] = len(quantized)

    return example

def load_data(cfg):
    """
    ljspeech 데이터를 hf에서 가져와서 전처리를 수행하는 함수
    데이터 전처리가 완료되면 데이터로더를 반환한다.

    Parameters:
    - cfg: 데이터 전처리 관련 하이퍼파라미터 config

    Returns:
    - train_loader: 학습용 데이터셋에 대한 데이터로더
    - test_loader: 검증용 데이터셋에 대한 데이터로더
    """
    ds = load_dataset('keithito/lj_speech', split="train")
    ds = ds.train_test_split(test_size=0.2, random_state=42)
    ds = ds.cast_column("audio", Audio(sampling_rate=cfg.data.sample_rate))
    ds = ds.with_format('torch')
    ds = ds.map(preprocess_audio, fn_kwargs={'cfg': cfg}, remove_columns=['audio', 'file', 'text', 'normalized_text'])

    train_set, test_set = ds['train'], ds['test']
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=cfg.test.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader

