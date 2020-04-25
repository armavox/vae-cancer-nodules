import argparse
import os
import warnings
import utils.helpers as H

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer

from lightning_boilerplates.lightning_crls import CRLSModel


warnings.filterwarnings("ignore", category=DeprecationWarning)


def run(config: argparse.Namespace):
    logger = TensorBoardLogger(
        save_dir=os.path.join(config.metaconf["ws_path"], 'tensorboard_logs'),
        name=config.metaconf["experiment_name"]
    )
    trainer = Trainer(
        gpus=config.metaconf["ngpus"],
        distributed_backend="dp",
        max_epochs=config.hyperparams["max_epochs"],
        logger=logger,
        # truncated_bptt_steps=10
    )

    # Start training
    model = CRLSModel(config)
    trainer.fit(model)


if __name__ == "__main__":
    args = H.arguments()
    config = H.load_params_namespace(args.train_config)
    use_cuda = (True if config.metaconf["ngpus"] != 0 else False)

    H.set_logging_config("./configs/logging_config.yaml", config.metaconf["ws_path"])
    H.random_seed_init(config.metaconf["random_seed"], use_cuda)

    run(config)
