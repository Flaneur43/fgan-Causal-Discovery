import torch

def acyclicity_loss(B):
    """
    NOTEARS-style acyclicity constraint:

        h(B) = tr(exp(B ∘ B)) - p
    """
    p = B.shape[0]
    return torch.trace(torch.matrix_exp(B * B)) - p

def bow_loss(B, Sigma):
    """
    Penalize simultaneous directed and bidirected edges.
    """
    p = B.shape[0]
    off_diagonal_mask = 1.0 - torch.eye(p, device=B.device)
    Sigma_offdiag = Sigma * off_diagonal_mask
    return torch.sum((B * B) * (Sigma_offdiag * Sigma_offdiag))

def simplicity_loss(Sigma):
    """
    Simple sparsity-style penalty.
    """
    return torch.sum(Sigma)