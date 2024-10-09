import hydra
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.ljspeech import load_data
from models.wavenet import WaveNet
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, ModelCheckpoint
from ignite.handlers.tensorboard_logger import TensorboardLogger, WeightsHistHandler
from ignite.metrics import Loss
from omegaconf import DictConfig
from pathlib import Path


def train_step(engine, batch, model, optimizer, criterion, device):
    model.train()
    inputs, targets, lengths = batch
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    return loss.item()


def val_step(engine, batch, model, criterion, device):
    model.eval()
    with torch.no_grad():
        inputs, targets, lengths = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.transpose(1, 2), targets)
    return loss.item()


def setup_engines(model, optimizer, criterion, device):
    trainer = Engine(lambda engine, batch: train_step(engine, batch, model, optimizer, criterion, device))
    evaluator = Engine(lambda engine, batch: val_step(engine, batch, model, criterion, device))
    return trainer, evaluator


def setup_logging(trainer, evaluator, model, cfg):
    # Tensorboard logger setup
    tb_logger = TensorboardLogger(log_dir=f"logs/{cfg.train.experiment_name}")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss}
    )

    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.ITERATION_COMPLETED,
        tag="validation",
        output_transform=lambda loss: {"loss": loss}
    )

    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=WeightsHistHandler(model)
    )

    @trainer.on(Events.INTERRUPT)
    @trainer.on(Events.TERMINATE)
    @trainer.on(Events.EXCEPTION_RAISED)
    def close_tb_logger(_):
        tb_logger.close()

    return tb_logger


def setup_checkpoints(trainer, model, optimizer, device, cfg):
    to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    checkpoint_handler = ModelCheckpoint(
        dirname=f'../checkpoints/{cfg.train.experiment_name}',
        filename_prefix='wavenet',
        n_saved=cfg.train.n_saved,
        create_dir=True,
        require_empty=False
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.train.ckpt_interval), checkpoint_handler, to_save)

    if cfg.train.resume_from != -1:
        checkpoint_fp = Path(cfg.train.resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location=device)
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and model
    train_loader, test_loader = load_data(cfg)
    model = WaveNet(cfg=cfg).to(device)

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Setup engines
    trainer, evaluator = setup_engines(model, optimizer, criterion, device)

    # Setup validation handler
    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.train.val_interval))
    def run_validation(_):
        evaluator.run(test_loader)

    # Attach loss metric to the evaluator
    Loss(criterion).attach(evaluator, "loss")

    # Setup checkpointing and resume
    setup_checkpoints(trainer, model, optimizer, device, cfg)

    # Attach progress bar
    ProgressBar().attach(trainer)

    # Setup tensorboard logging
    setup_logging(trainer, evaluator, model, cfg)

    # Run training
    trainer.run(train_loader, max_epochs=cfg.train.total_epochs)


if __name__ == '__main__':
    main()
