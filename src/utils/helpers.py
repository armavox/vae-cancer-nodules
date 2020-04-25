import json
import logging
import logging.config
import math
import matplotlib.pyplot as plt
import os
import re
import yaml
from argparse import ArgumentParser, Namespace
from datetime import datetime
from matplotlib.lines import Line2D
from typing import Iterable

import numpy as np
import torch
from torch.autograd import grad


log = logging.getLogger("utils.helpers")


class LoggingFilter(logging.Filter):
    def filter(self, record):
        allow = record.name in logging_config["loggers"]
        return allow


def arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--train-config", type=str, help="Path to the yaml file with the training parameters"
    )
    return parser.parse_args()


def config_snapshot(name: str, config: dict, old_config_path: str):
    if os.path.exists(old_config_path):
        with open(old_config_path) as f:
            old_config = json.load(f)
        shared_items = {
            k: old_config[k] for k in old_config if k in config and old_config[k] == config[k]
        }
        if len(shared_items) == len(config):
            return True
        log.warning(f"{name} configs are not similar. Snapshot refreshed")
    log.warning(f"{name} config file doesn't exist. Snapshot created")
    with open(old_config_path, "w") as f:
        json.dump(config, f)
    return False


def get_gradient_norm(parameters: Iterable[torch.nn.parameter.Parameter], norm_type=2) -> float:
    if norm_type == math.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
    return grads


def load_params_namespace(yaml_path: str) -> Namespace:
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
                   [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                   |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                   |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                   |[-+]?\\.(?:inf|Inf|INF)
                   |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with open(yaml_path) as config_file:
        config = yaml.load(config_file, Loader=loader)
        return Namespace(**config)


def make_np(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    x = x.cpu().numpy()
    return x


def makedirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def plot_grad_flow(named_parameters, legend_model_name, legend_epoch, savepath, return_fig=True):
    """Plots the gradients flowing through different layers
    in the net during training. Can be used for checking for
    possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after
    loss.backward() as  "plot_grad_flow(self.model.named_parameters())"
    to visualize the gradient flow.
    """

    ave_grads, max_grads, layers = [], [], []

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    fig = plt.figure(figsize=(12, 3), dpi=90)
    plt.barh(np.arange(1, len(max_grads) + 1), max_grads, alpha=0.4, height=0.5, color="c")
    plt.barh(np.arange(1, len(max_grads) + 1), ave_grads, alpha=0.4, height=0.5, color="b")
    plt.xscale("log")

    plt.vlines(0, 0, len(ave_grads) + 1, lw=1, color="k")
    plt.yticks(range(1, len(ave_grads) + 1, 1), layers)
    plt.ylim(0, len(ave_grads) + 1)
    max_grad = float(torch.max(torch.stack(max_grads)))
    plt.xlim(0, 1.2 * (max_grad if not np.isnan(max_grad) else 1))
    plt.ylabel("Layers")

    plt.title(f"{legend_model_name}. Epoch {legend_epoch}. Gradient flow")
    plt.grid(alpha=0.4)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
        loc=1,
    )
    plt.tight_layout()
    filename = f"{legend_model_name}_{legend_epoch}"
    plt.savefig(os.path.join(savepath, filename))
    if return_fig:
        return fig
    else:
        plt.close()


def random_seed_init(random_seed: bool = None, cuda: bool = False):
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_logging_config(logging_config_path: str, ws_path: str):
    with open(logging_config_path, "r") as f:
        global logging_config
        logging_config = yaml.safe_load(f.read())
        now = datetime.now().strftime("%Y-%m-%d-%H:%M")
        logdir = os.path.join(ws_path, "artifacts", "logs")
        makedirs(logdir)
        logging_config["handlers"]["file"]["filename"] = f"{logdir}/{now}.log"
        logging.config.dictConfig(logging_config)


def train_val_holdout_split(dataset, ratios=[0.7, 0.2, 0.1]):
    """Return indices for subsets of the dataset.
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset made with class which inherits `torch.utils.data.Dataset`
    ratios : list of floats
        List of [train, val, holdout] ratios respectively. Note, that sum of
        values must be equal to 1. (train + val + holdout = 1.0)
    """

    assert np.allclose(ratios[0] + ratios[1] + ratios[2], 1)
    train_ratio, val_ratio, test_ratio = ratios

    df_size = len(dataset)
    train_inds = np.random.choice(range(df_size), size=int(df_size * train_ratio), replace=False)
    val_test_inds = list(set(range(df_size)) - set(train_inds))
    val_inds = np.random.choice(
        val_test_inds,
        size=int(len(val_test_inds) * val_ratio / (val_ratio + test_ratio)),
        replace=False,
    )

    test_inds = np.asarray(list(set(val_test_inds) - set(val_inds)), dtype="int")

    assert len(list(set(train_inds) - set(val_inds) - set(test_inds))) == len(train_inds)

    return train_inds, val_inds, test_inds
