from functools import partial

import numpy as np
import raster_geometry
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid

from data.transforms import img_derivative, heaviside, SOBEL_X, SOBEL_Y


class RLSModule(nn.Module):
    """RLS module

    Parameters
    ----------
    input_size : tuple
        Input image size
    hidden_feat : int
        [description]
    out_feat : int
        [description]

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
    """

    def __init__(self, input_size: tuple):
        super().__init__()

        self.img_size = input_size[-2:]  # [H, W]
        in_feat = np.prod(self.img_size)

        self.ug = nn.Parameter(torch.zeros(in_feat, in_feat).normal_(std=0.01))
        self.wg = nn.Parameter(torch.zeros(in_feat, in_feat).normal_(std=0.01))
        nn.init.eye_(self.ug)
        nn.init.eye_(self.wg)

        # self.uzr = nn.Parameter(torch.zeros(2 * in_feat, in_feat).normal_(std=0.01))
        # self.wzr = nn.Parameter(torch.zeros(2 * in_feat, in_feat).normal_(std=0.01))
        # self.bz = nn.Parameter(torch.zeros(2 * in_feat))
        # self.uo = nn.Parameter(torch.zeros(in_feat, in_feat).normal_(std=0.01))
        # self.wo = nn.Parameter(torch.zeros(in_feat, in_feat).normal_(std=0.01))
        # self.bo = nn.Parameter(torch.zeros(in_feat))

        self.gru = nn.GRUCell(in_feat, in_feat)
        self.dense = nn.Linear(in_feat, in_feat)

        nn.init.eye_(self.gru.weight_hh)
        nn.init.eye_(self.gru.weight_ih)
        self.gru.bias_ih.data.zero_()
        self.gru.bias_hh.data.zero_()

        nn.init.xavier_normal_(self.dense.weight)
        self.dense.bias.data.zero_()

        self.writer = None

    def forward(self, input, hidden, writer: SummaryWriter = None, step: int = None):
        """input: image [B, C, H, W], hidden: level set [B, C, H, W]; C == 1"""

        batch_size = input.size(0)

        c1, c2 = self.avg_inside(input, hidden.detach()), self.avg_outside(input, hidden.detach())
        I_c1 = ((input - c1) ** 2).view(batch_size, -1)
        I_c2 = ((input - c2) ** 2).view(batch_size, -1)
        kappa = self.curvature(hidden).view(batch_size, -1)
        make_grid_p = partial(make_grid, nrow=4, normalize=True)
        if writer is not None:
            writer.add_image(f"rls/hidden", make_grid_p(hidden.detach().view(-1, 1, 64, 64)), step)
            writer.add_image(f"rls/I_c1", make_grid_p(I_c1.detach().view(-1, 1, 64, 64)), step)
            writer.add_image(f"rls/I_c2", make_grid_p(I_c2.detach().view(-1, 1, 64, 64)), step)
            writer.add_image(f"rls/kappa", make_grid(kappa.detach().view(-1, 1, 64, 64), 4), step)

        x = kappa - torch.mm(I_c1, self.ug) + torch.mm(I_c2, self.wg)
        # hidden = gru_rls_cell(
        #   x, hidden.view(batch_size, -1), self.uzr, self.wzr, self.bz, self.uo, self.wo, self.bo
        # )
        hidden = self.gru(x, hidden.view(batch_size, -1))
        hidden = F.relu(hidden)
        output = self.dense(hidden)
        return (
            output.view(batch_size, *self.img_size),
            hidden.view(batch_size, 1, *self.img_size),
        )

    def curvature(self, hidden):
        """hidden.size: [B, C, H, W], C == 1"""

        phi_dx = img_derivative(hidden, SOBEL_X)
        phi_dxx = img_derivative(phi_dx, SOBEL_X)
        phi_dy = img_derivative(hidden, SOBEL_Y)
        phi_dyy = img_derivative(phi_dy, SOBEL_Y)
        phi_dxy = img_derivative(phi_dx, SOBEL_Y)

        kappa_nom = (phi_dxx * phi_dy ** 2 - 2 * phi_dx * phi_dy * phi_dxy + phi_dyy * phi_dx ** 2)
        kappa_denom = (phi_dx ** 2 + phi_dy ** 2) ** (3 / 2) + 1e-8
        kappa = kappa_nom / kappa_denom
        return kappa

    def avg_inside(self, input, level_set):
        mask = heaviside(level_set) > 0.5
        inside = input * mask
        return inside.mean()

    def avg_outside(self, input, level_set):
        mask = heaviside(level_set) > 0.5
        outside = input * ~mask
        return outside.mean()


def init_levelset(img_size: tuple, shape: str = "checkerboard"):
    """Levelset init"""

    level_set = torch.zeros(img_size)
    if shape == "checkerboard":
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                level_set[i, j] = np.sin(i * np.pi / 10) * np.sin(j * np.pi / 10)
    elif shape == "circle":
        mask = raster_geometry.circle(img_size, img_size[0] // 2 - 5)
        level_set[mask] = 1
    else:
        raise NotImplementedError
    return level_set


# import matplotlib.pyplot as plt
# plt.imshow((hidden.cpu().detach().numpy()[0][0]))
# plt.colorbar()
# plt.savefig('hidden.png')
# plt.close()
