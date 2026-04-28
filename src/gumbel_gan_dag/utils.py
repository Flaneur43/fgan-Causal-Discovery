
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_weights_B(p, device=None):
    """
    Sample random weights for directed edges.
    Weights are from ±U(0.5, 2.0).
    """
    weights_abs = torch.rand(p, p, device=device) * 1.5 + 0.5
    signs = torch.sign(torch.rand(p, p, device=device) - 0.5)
    return weights_abs * signs

def sample_weights_Sigma(p, device=None):
    """
    Sample random weights for Sigma.
    Off-diagonal: ±U(0.4, 0.7)
    Diagonal: U(0.7, 1.2)
    """
    tril_indices = torch.tril_indices(row=p, col=p, offset=-1, device=device)
    diag_indices = torch.arange(p, device=device).repeat(2, 1)

    weights_Sigma_tril = torch.zeros(p, p, device=device)

    off_diag_abs = torch.rand(p, p, device=device) * 0.3 + 0.4
    off_diag_signs = torch.sign(torch.rand(p, p, device=device) - 0.5)
    off_diag_weights = off_diag_abs * off_diag_signs

    weights_Sigma_tril[tril_indices[0], tril_indices[1]] = off_diag_weights[
        tril_indices[0],
        tril_indices[1],
    ]

    diag_weights = torch.rand(p, device=device) * 0.5 + 0.7
    weights_Sigma_tril[diag_indices[0], diag_indices[1]] = diag_weights

    weights_Sigma = (
        weights_Sigma_tril
        + weights_Sigma_tril.T
        - torch.diag(torch.diag(weights_Sigma_tril))
    )

    return weights_Sigma

def get_prob_B(generator):
    return torch.sigmoid(generator.logits_B)

def get_prob_Sigma(generator):
    p = generator.p

    logits_Sigma_tril = torch.zeros(
        p,
        p,
        device=generator.logits_Sigma_params.device,
    )

    tril_indices = generator.tril_indices
    logits_Sigma_tril[tril_indices[0], tril_indices[1]] = generator.logits_Sigma_params

    logits_Sigma = (
        logits_Sigma_tril
        + logits_Sigma_tril.T
        - torch.diag(torch.diag(logits_Sigma_tril))
    )

    return torch.sigmoid(logits_Sigma)

def threshold_structure(prob_matrix, ratio=0.7):
    threshold = prob_matrix.max() * ratio
    return (prob_matrix > threshold).float().cpu().numpy()

def get_true_structures(B_true, Sigma_e_true):
    B_true_structure = (np.abs(B_true) > 1e-6).astype(int)

    Sigma_true_structure = (np.abs(Sigma_e_true) > 1e-6).astype(int)
    np.fill_diagonal(Sigma_true_structure, 1)

    return B_true_structure, Sigma_true_structure