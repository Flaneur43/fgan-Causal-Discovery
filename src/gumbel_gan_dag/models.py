import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.p = p

        self.logits_B = nn.Parameter(torch.zeros(p, p, dtype=torch.float32))

        num_tril_elements = p * (p - 1) // 2
        self.logits_Sigma_params = nn.Parameter(
            torch.zeros(num_tril_elements, dtype=torch.float32)
        )

        self.register_buffer(
            "tril_indices",
            torch.tril_indices(row=p, col=p, offset=-1),
        )

        self.register_buffer(
            "diag_indices",
            torch.arange(p).repeat(2, 1),
        )

    def forward(self, Z, weights_B, weights_Sigma, tau=1.0):
        p = self.p
        device = Z.device

        # Directed edge mask
        logits_B_binary = torch.stack(
            [torch.zeros_like(self.logits_B), self.logits_B],
            dim=-1,
        )

        soft_B = F.gumbel_softmax(
            logits_B_binary,
            tau=tau,
            hard=False,
        )[..., 1]

        # Bidirected edge mask
        logits_Sigma_binary = torch.stack(
            [torch.zeros_like(self.logits_Sigma_params), self.logits_Sigma_params],
            dim=-1,
        )

        soft_Sigma_tril = F.gumbel_softmax(
            logits_Sigma_binary,
            tau=0.1,
            hard=False,
        )[..., 1]

        temp = torch.zeros(p, p, device=device)
        temp[self.tril_indices[0], self.tril_indices[1]] = soft_Sigma_tril

        soft_Sigma = temp + temp.T + torch.eye(p, device=device)

        B_star = soft_B * weights_B
        Sigma_star = soft_Sigma * weights_Sigma

        # Make Sigma_star positive definite if necessary
        eigenvalues = torch.linalg.eigh(Sigma_star)[0].detach()
        min_eig = -torch.logsumexp(-100 * eigenvalues, dim=-1) / 100

        if min_eig < 0:
            Sigma_star = Sigma_star + (-min_eig + 1e-4) * torch.eye(p, device=device)

        success = True

        try:
            L_star = torch.linalg.cholesky(Sigma_star)
        except torch._C._LinAlgError:
            success = False
            L_star = torch.eye(p, device=device)

        E = Z @ L_star.T

        identity = torch.eye(p, device=device)

        try:
            inv_I_minus_B = torch.inverse(identity - B_star)
        except torch._C._LinAlgError:
            success = False
            inv_I_minus_B = torch.eye(p, device=device)

        X_fake = E @ inv_I_minus_B

        return X_fake, soft_B, soft_Sigma, soft_Sigma_tril, success

class Discriminator(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(p, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)