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

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
