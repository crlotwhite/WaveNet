import hydra
import torch

from dataset.ljspeech import load_data
from omegaconf import DictConfig

@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    train_loader, test_loader = load_data(cfg)

    for batch in train_loader:
        print(torch.sum(batch[0]))


if __name__ == '__main__':
    main()