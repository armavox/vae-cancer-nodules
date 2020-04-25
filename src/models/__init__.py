# Code provided by:
# @misc{Subramanian2020,
#   author = {Subramanian, A.K},
#   title = {PyTorch-VAE},
#   year = {2020},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}}
# }

from .base import BaseVAE, Tensor
from .vanilla_vae import VanillaVAE
from .cvae import ConditionalVAE


vae_models = {
    "VanillaVAE": VanillaVAE,
    "ConditionalVAE": ConditionalVAE
}


def get_vae_model(name: str) -> BaseVAE:
    if name in vae_models:
        return eval(name)
    else:
        raise NotImplementedError
