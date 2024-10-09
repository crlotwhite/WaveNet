import hydra
import numpy as np
import os
import soundfile as sf
import torch
import torch.nn.functional as F

from models.wavenet import WaveNet


def preprocess_initial_input(initial_value, num_classes, device):
    """
    초기 입력을 준비하는 함수. 초기 입력은 one-hot 형태로 설정합니다.

    Parameters:
    - initial_value: 초기 샘플 값 (0부터 num_classes-1 범위의 정수 값, 보통 중간값으로 설정)
    - num_classes: 클래스의 개수 (예: 256, μ-law 양자화된 값의 개수)
    - device: 모델이 위치한 장치 ('cpu' 또는 'cuda')

    Returns:
    - 초기 입력 텐서 (1, num_classes, 1) 형태
    """
    initial_input = torch.zeros(1, num_classes, 1).to(device)
    initial_input[0, initial_value, 0] = 1  # one-hot 벡터 생성
    return initial_input


def generate_audio(model, initial_input, num_samples, device):
    """
    학습된 WaveNet 모델을 사용하여 오디오 샘플을 생성하는 함수.

    Parameters:
    - model: 학습된 WaveNet 모델
    - initial_input: 초기 입력 (batch_size=1, num_classes, 1) 형태
    - num_samples: 생성할 샘플의 개수
    - device: 모델이 위치한 장치 ('cpu' 또는 'cuda')

    Returns:
    - generated_audio: μ-law 양자화된 값으로 생성된 오디오 시퀀스 (numpy 배열)
    """
    model.eval()  # 모델을 평가 모드로 전환
    generated = [initial_input]

    with torch.no_grad():
        for _ in range(num_samples):
            # 현재까지 생성된 데이터를 모델에 입력
            input_tensor = torch.cat(generated, dim=2)  # (1, num_classes, current length)
            output = model(input_tensor)  # 모델의 출력: (1, num_classes, current length)

            # 마지막 타임스텝에 대한 logits을 가져옴
            logits = output[:, :, -1]

            # 소프트맥스 적용하여 확률 분포 얻기
            probabilities = F.softmax(logits, dim=1).squeeze()

            # 확률 분포에 따라 다음 샘플 생성
            next_sample = torch.multinomial(probabilities, num_samples=1).unsqueeze(-1)

            # 생성된 샘플을 원-핫 벡터 형태로 변환
            one_hot_next_sample = F.one_hot(next_sample, num_classes=initial_input.size(1)).permute(0, 2, 1).float()
            generated.append(one_hot_next_sample)

    # 최종적으로 생성된 오디오의 μ-law 양자화된 값 반환
    generated_audio = torch.cat(generated, dim=2).argmax(dim=1).squeeze().cpu().numpy()
    return generated_audio


def inverse_mulaw(quantized, mu=255):
    """
    μ-law 압축된 오디오 데이터를 원래의 부동소수점 값으로 복원합니다.

    Parameters:
    - quantized: μ-law 양자화된 오디오 데이터 (정수 형태의 numpy 배열)
    - mu: μ-law 파라미터 (기본값: 255, 8비트 양자화 기준)

    Returns:
    - waveform: 원래의 부동소수점 오디오 신호 (numpy 배열, -1 ~ 1 사이 값)
    """
    quantized = quantized.astype(np.float32)
    quantized = 2 * (quantized / mu) - 1.0
    waveform = np.sign(quantized) * (1 / mu) * ((1 + mu) ** np.abs(quantized) - 1)
    return waveform

@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드 및 설정
    model = WaveNet(cfg).to(device)

    # 가장 최근에 저장된 체크포인트로부터 로드
    ckpt_dir_list = os.listdir(f'../checkpoints/{cfg.train.experiment_name}')
    ckpt = torch.load(ckpt_dir_list[-1])
    model.load_state_dict(ckpt['model'])

    # 초기 입력 준비 (μ-law 양자화의 중간값으로 설정)
    initial_input = preprocess_initial_input(initial_value=127, num_classes=256, device=device)

    # 오디오 샘플 생성
    num_samples = 16000  # 예를 들어, 1초 길이의 오디오 생성 (16kHz 샘플링)
    generated_audio = generate_audio(model, initial_input, num_samples, device)

    # μ-law 복원
    waveform = inverse_mulaw(generated_audio)

    # 오디오 신호를 파일로 저장
    sf.write('generated_audio.wav', waveform, samplerate=16000)


if __name__ == "__main__":
    main()
