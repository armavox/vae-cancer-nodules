from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import torch


SOBEL_X = (
    torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
)

SOBEL_Y = (
    torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
)


class CustomTransformNumpy(ABC):
    """Abstract method for custom numpy transformations.
    Every subclass should implement `__init__` for
    transformations parameters setting and `__call__` method for application to image.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Normalization(CustomTransformNumpy):
    def __init__(
        self,
        sample: List[np.ndarray] = None,
        from_min: float = None,
        from_max: float = None,
        to_min: float = None,
        to_max: float = None,
    ):
        self.to_min, self.to_max = to_min, to_max
        self.to_span = self.to_max - self.to_min

        if sample:
            sample = np.concatenate(sample)
            self.from_min = np.min(sample)
            self.from_max = np.max(sample)
        else:
            assert (from_min is not None) and (from_max is not None)
            self.from_min = from_min
            self.from_max = from_max
        self.from_span = self.from_max - self.from_min

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """ min max normalization"""
        scaled = (volume - self.from_min) / self.from_span
        return scaled * self.to_span + self.to_min

    def denorm(self, volume: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Denormalization with pre-saved stats"""
        scaled = (volume - self.to_min) / self.to_span
        return scaled * self.from_span + self.from_min


class Rot90:
    def __init__(self, num_rots=None):
        if num_rots:
            self.num_rots = num_rots
        else:
            self.num_rots = np.random.randint(10)

    def __call__(self, input):
        input = input.squeeze()
        rot = torch.rot90(input, self.num_rots, (0, 1))
        return rot.unsqueeze(0)


class Flip:
    def __init__(self, flip_type: str = None):
        if flip_type:
            self.type = flip_type
        else:
            self.type = np.random.choice(["h", "v"])

    def __call__(self, input):
        if self.type == "h":
            flipped = torch.flip(input, (0, 1))
        elif self.type == "v":
            flipped = torch.flip(input, (0, 2))
        else:
            raise NotImplementedError
        return flipped


class AugmentationNoduleDict(ABC):
    def __init__(self, autoupd=True):
        self.autoupd = autoupd

    def __call__(self, nodule: torch.Tensor) -> torch.Tensor:
        """Transform data

        Parameters
        ----------
        nodule : torch.Tensor, [C D H W]
            input tensor

        Returns
        -------
        torch.Tensor, [C D H W]
        """

        tensor = self._augmentation(nodule)
        if self.autoupd:
            self._update()
        return tensor

    @abstractmethod
    def _augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _update(self):
        pass


class RotNodule3D(AugmentationNoduleDict):
    def __init__(self, dims_pair: list = None, autoupd=True):
        super().__init__(autoupd)
        self._dims_pair = dims_pair
        if dims_pair is None:
            self._dims_pair = [[1, 2], [1, 3], [2, 3]]
        else:
            self._dims_pair = [dims_pair]
        self._count_dims = len(self._dims_pair)
        self._count_rotate = [0] * self._count_dims

    def _augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        for i in range(self._count_dims):
            dims = self._dims_pair[i]
            count_rotate = self._count_rotate[i]
            if count_rotate == 0:
                continue

            tensor = torch.rot90(tensor, count_rotate, dims)

        return tensor

    def _update(self):
        self._count_rotate = np.random.randint(-3, 4, self._count_dims).tolist()


class FlipNodule3D(AugmentationNoduleDict):
    def __init__(self, dims: list = None, autoupd=True):
        super().__init__(autoupd)
        if dims is None:
            self._dims = [1, 2, 3]
        else:
            self._dims = dims
        self._count_dims = len(self._dims)
        self._need_flip = [0] * self._count_dims

    def _augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        for i in range(self._count_dims):
            dim = self._dims[i]
            need_flip = self._need_flip[i]
            if need_flip == 0:
                continue

            tensor = torch.flip(tensor, [dim])

        return tensor

    def _update(self):
        self._need_flip = np.random.randint(0, 2, self._count_dims).tolist()


class TranslateNodule3D(AugmentationNoduleDict):
    def __init__(self, dims: list = None, shift_val: int = 10, autoupd=True):
        super().__init__(autoupd)
        if dims is None:
            self._dims = [1, 2, 3]
        else:
            self._dims = dims
        self._count_dims = len(self._dims)
        self._shifts = [0] * self._count_dims
        self._shift_val = shift_val

    def _augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.roll(tensor, self._shifts, self._dims)
        if self._shifts[0] < 0:
            tensor[:, :, tensor.size()[1] + self._shifts[0] :] = 0
        elif self._shifts[0] > 0:
            tensor[:, :, : self._shifts[0]] = 0

        if self._shifts[1] < 0:
            tensor[:, :, :, tensor.size()[2] + self._shifts[1] :] = 0
        elif self._shifts[1] > 0:
            tensor[:, :, :, : self._shifts[1]] = 0

        if self._shifts[2] < 0:
            tensor[:, :, :, :, tensor.size()[3] + self._shifts[2] :] = 0
        elif self._shifts[2] > 0:
            tensor[:, :, :, :, : self._shifts[2]] = 0

        return tensor

    def _update(self):
        self._shifts = np.random.randint(-self._shift_val, self._shift_val, self._count_dims).tolist()
        self._shifts[2] = 0


class CropCenterNodule3D(AugmentationNoduleDict):
    def __init__(self, final_size: float):
        super().__init__()
        self.final_size = final_size

    def _augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        sizes_per_dim = np.array(tensor.size()[1:])
        sizes_per_dim -= self.final_size
        sizes_per_dim //= 2
        sizes_per_dim_shifted = sizes_per_dim + self.final_size

        return tensor[
            :,
            sizes_per_dim[0] : sizes_per_dim_shifted[0],
            sizes_per_dim[1] : sizes_per_dim_shifted[1],
            sizes_per_dim[2] : sizes_per_dim_shifted[2],
        ]

    def _update(self):
        pass


def heaviside(mask, eps=1e-5):
    return 1 / 2 * (1 + (2 / np.pi) * (torch.atan(mask / eps)))


def img_derivative(input: torch.FloatTensor, sobel_kernel: torch.FloatTensor) -> torch.FloatTensor:
    assert input.dim() == 4
    assert sobel_kernel.dim() == 4
    conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv.weight = torch.nn.Parameter(sobel_kernel.type_as(input), requires_grad=False)
    return conv(input)  # [N, C, H, W]
