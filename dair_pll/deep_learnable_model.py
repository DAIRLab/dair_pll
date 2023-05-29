from abc import ABC, abstractmethod
from typing import Type, cast, List, Tuple, Callable

import torch
from torch import Tensor, nn
from torch.nn import Module, Parameter


class DeepLearnableModel(ABC, Module):
    mean: Parameter
    std_dev: Parameter

    def __init__(self, in_size: int, _hidden_size: int, _out_size: int,
                 _layers: int, _nonlinearity: Type[Module]):
        super().__init__()
        self.mean = Parameter(torch.ones(in_size), requires_grad=False)
        self.std = Parameter(torch.ones(in_size), requires_grad=False)

    @abstractmethod
    def sequential_eval(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    def set_normalization(self, x: Tensor) -> None:
        # flatten x into shape num_points * in_size
        x = torch.flatten(x, end_dim=-2)
        self.mean = Parameter(x.mean(dim=0), requires_grad=False)
        self.std = Parameter(x.std(dim=0), requires_grad=False)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class DeepRecurrentModel(DeepLearnableModel):
    encoder: Callable[[Tensor], Tensor]

    def __init__(self, in_size: int, hidden_size: int, out_size: int,
                 layers: int, nonlinearity: Type[Module]) -> None:
        super().__init__(in_size, hidden_size, out_size, layers, nonlinearity)
        encode = True
        if encode:
            self.encoder = _mlp(in_size, hidden_size, hidden_size, layers // 2,
                                nonlinearity)
        else:
            self.encoder = lambda x: cast(Tensor, x)
        self.decoder = _mlp(hidden_size, hidden_size, out_size,
                            layers - (layers // 2), nonlinearity)
        rnn_in_size = hidden_size if encode else in_size
        self.recurrent = nn.GRU(input_size=rnn_in_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                batch_first=True)

    def forward(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        (next_recurrent_output, carry) = self.sequential_eval(x, carry)
        return self.decoder(next_recurrent_output), carry

    def sequential_eval(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        # x is B x L x N
        # carry is B x H
        carry = carry.transpose(0, 1)
        for i in range(x.shape[1]):
            xi = self.normalize(x[:, i:(i + 1), :])
            recurrent_output, carry = self.recurrent(self.encoder(xi), carry)
        return recurrent_output, carry.transpose(0, 1)


def _mlp(in_size: int, hidden_size: int, out_size: int, layers: int,
         nonlinearity: Type[Module]) -> Module:
    modules = cast(List[Module], [])
    if layers == 0:
        return nn.Linear(in_size, out_size)
    modules.append(nn.Linear(in_size, hidden_size))
    for i in range(layers - 1):
        modules.append(nonlinearity())
        modules.append(nn.Linear(hidden_size, hidden_size))
    modules.append(nonlinearity())
    modules.append(nn.Linear(hidden_size, out_size))
    return nn.Sequential(*modules)


class MLP(DeepLearnableModel):
    net: Module

    def __init__(self, in_size: int, hidden_size: int, out_size: int,
                 layers: int, nonlinearity: Type[Module]) -> None:
        super().__init__(in_size, hidden_size, out_size, layers, nonlinearity)
        self.net = _mlp(in_size, hidden_size, out_size, layers, nonlinearity)

    def forward(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        return self.net(self.normalize(x)), carry

    def sequential_eval(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        # x is B x L x N
        return self.net(self.normalize(x[:, -1, :])).unsqueeze(-2), carry


class ZeroModel(DeepLearnableModel):

    def __init__(self, in_size: int, hidden_size: int, out_size: int,
                 layers: int, nonlinearity: Type[Module]) -> None:
        super().__init__(in_size, hidden_size, out_size, layers, nonlinearity)
        self.out_size = out_size
        self.dummy_param = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        return self.sequential_eval(x, carry)

    def sequential_eval(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        # x is B x L x N
        return self.dummy_param * torch.zeros(
            (x.shape[0], 1, self.out_size)), carry
