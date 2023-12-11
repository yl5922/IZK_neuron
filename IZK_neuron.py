from spikingjelly.clock_driven import neuron, surrogate
from typing import Callable, overload
import torch
import torch.nn as nn
class IZK_neuron(nn.Module):
    def __init__(self, v_threshold: float = -4.,
                 v_reset: float = -6.5, surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__()
        self.a = 0.02
        self.b = 0.2
        self.c = v_reset
        self.d = 8
        self.v = self.c
        self.mu = self.b * self.c
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function

    def neuronal_charge(self, x: torch.Tensor):
        # self.v = self.v + 1 * (0.4 * self.v ** 2 + 5 * self.v + 14 - self.mu + x)
        # self.mu = self.mu + self.a*(self.b*self.v - self.mu)

        #conditional IZK
        # flag = self.v < -7.76
        # tmp_posi = self.v + 1 * (0.4 * self.v * self.v + 5 * self.v + 14.0 - self.mu + x)
        # tmp_nega = self.v - (self.v - self.v_reset) / 2 - 1.0*self.mu + 1.0*x
        # tmp_posi[flag] = tmp_nega[flag]
        # self.v = tmp_posi
        #self.mu = self.mu + self.a * (self.b * self.v - self.mu)
        #LIF
        self.v = self.v - (self.v - self.v_reset) / 2 + x

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        spike_d = spike.detach()

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset
            #self.mu = self.mu + self.d/10

    def forward(self, x):
        self.initialize(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def initialize(self, x):
        "Initialize the neuron parameters based on the dimension of the input"
        if isinstance(self.v, float):
            v_init = self.v
            mu_init = self.mu
            self.v = torch.full_like(x.data, v_init)
            self.mu = torch.full_like(x.data, mu_init)

    def reset(self):
        self.v = self.c
        self.mu = self.b * self.c



