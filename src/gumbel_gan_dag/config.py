
import math
from dataclasses import dataclass

@dataclass
class Config:
    p: int = 4
    n_samples: int = 2000

    lr_d: float = 0.0001
    lr_g: float = 0.001

    lambda_acyc: float = 1.0
    lambda_bow: float = 1.0
    lambda_simple: float = 1.0

    num_epochs: int = 4000
    batch_size: int = 10

    tau_start: float = 1.0
    tau_end: float = 0.1

    @property
    def anneal_rate(self):
        return math.exp(
            math.log(self.tau_end / self.tau_start) / (self.num_epochs / 4)
        )