from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F

from models import BaseVAE


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(flat_latents ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss + self.beta * embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dims: List = None,
        n_residual_layers: int = 6,
        beta: float = 0.25,
        img_size: int = 64,
        **kwargs,
    ) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        # modules = []
        # if hidden_dims is None:
        #     hidden_dims = [32, 64]

        # # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
        #             nn.LeakyReLU(),
        #         )
        #     )
        #     in_channels = h_dim

        # modules.append(
        #     nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),)
        # )

        # for _ in range(n_residual_layers):
        #     modules.append(ResidualLayer(in_channels, in_channels))
        # modules.append(nn.LeakyReLU())

        # modules.append(nn.Sequential(nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),))

        # self.encoder = nn.Sequential(*modules)

        # self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # # Build Decoder
        # modules = []
        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
        #     )
        # )

        # for _ in range(n_residual_layers):
        #     modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        # modules.append(nn.LeakyReLU())

        # hidden_dims.reverse()
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(
        #                 hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1
        #             ),
        #             nn.LeakyReLU(),
        #         )
        #     )

        # modules.append(
        #     nn.Sequential(
        #         nn.ConvTranspose2d(
        #             hidden_dims[-1], out_channels=out_channels, kernel_size=4, stride=2, padding=1
        #         ),
        #         # nn.Tanh(),
        #     )
        # )

        # self.decoder = nn.Sequential(*modules)
        self.encoder = Encoder(1, h_dim=128, n_res_layers=2, res_h_dim=128, embedding_dim=64)
        self.vq_layer = VectorQuantizer2(n_e=512, e_dim=64, beta=.25)
        self.decoder = Decoder(in_dim=64, h_dim=128, n_res_layers=4, res_h_dim=128)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.Tensor) [B x D x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: torch.Tensor, return_embed: bool = False, **kwargs) -> List[torch.Tensor]:
        encoding = self.encode(input)[0]
        if return_embed:
            return encoding.squeeze()
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "VQ_Loss": vq_loss}

    def sample(self, num_samples: int, current_device: Union[int, str], **kwargs) -> torch.Tensor:
        raise Warning("VQVAE sampler is not implemented.")

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def embed(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        return self.forward(input, return_embed=True, **kwargs)


class ResidualLayer2(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer2(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class VectorQuantizer2(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        device = z.device
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.
    For a VQ VAE, q_theta outputs parameters of a categorical distribution.
    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, embedding_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.pre_quantization_conv(x)
        return x


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.
    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 1, kernel_size=kernel, stride=stride, padding=1),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
