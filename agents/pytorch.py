import torch
import torch.nn as nn

import numpy as np
import toolz

from typing import Literal

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print(f"Device is: {device}")


class LinearTorchPolicy(nn.Module):
    def __init__(self, dimensions):
        super(LinearTorchPolicy, self).__init__()

        dims = toolz.sliding_window(2, dimensions)
        linears = [torch.nn.Linear(in_dim, out_dim) for in_dim, out_dim in dims]
        activations = [nn.ReLU() for _ in range(len(linears) - 1)]

        net = toolz.interleave([linears, activations])

        self.net = torch.nn.Sequential(*net).to(device)

    def forward(self, x):
        action_values = self.net(x)

        return action_values


class TorchPolicyAgent:
    def __init__(self, policy: torch.nn.Module, mode: Literal["deterministic", "probabilistic"] = "deterministic"):
        self.policy = policy
        self.mode = mode

        self.parameters = self.policy.parameters

    def act(self, observation: np.ndarray):
        action_values = self.policy.forward(torch.Tensor(observation).to(device))

        if self.mode == "deterministic":
            action = torch.argmax(action_values).item()
        elif self.mode == "probabilistic":
            raise NotImplemented(f"No implemenation for mode: {self.mode}")
        else:
            raise Exception("Unrecognised mode: " + self.mode)

        return action


@toolz.curry
def add_gaussian_noise(noise_strength, model):
    genome = nn.utils.parameters_to_vector(model.parameters())
    noise = torch.randn(genome.shape).to(device) * noise_strength

    genome += noise
    torch.nn.utils.vector_to_parameters(genome, model.parameters())


