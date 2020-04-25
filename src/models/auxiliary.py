import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.0)
        init.constant(self.update_gate.bias, 0.0)
        init.constant(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).cuda()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


def gru_rls_cell(input, hidden, uzr, wzr, bz, uo, wo, bo):

    gates = F.linear(input, uzr) + F.linear(hidden, wzr, bz)
    resetgate, updategate = gates.chunk(2, 1)
    resetgate = torch.sigmoid(resetgate)
    updategate = torch.sigmoid(updategate)

    outgate = F.linear(input, uo) + F.linear(hidden * resetgate, wo, bo)
    outgate = torch.tanh(outgate)

    return updategate * hidden + (1 - updategate) * outgate


def lstm_cell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

    hx, cx = hidden  # w_ih: (256, 4), b_ih: (256); w_hh: (256, 64), b_hh: (256)
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy
