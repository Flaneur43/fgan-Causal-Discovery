# data generation
import numpy as np

def generate_random_dag(p, sparsity=1.0):
    """
    Generate a random weighted DAG adjacency matrix.
    """
    B = np.zeros((p, p), dtype=np.float32)
    order = np.random.permutation(p)

    for i in range(p):
        for j in range(i + 1, p):
            if np.random.rand() < sparsity:
                weight = (
                    np.random.uniform(0.5, 2.0)
                    if np.random.rand() > 0.5
                    else np.random.uniform(-2.0, -0.5)
                )
                B[order[i], order[j]] = weight

    return B

def generate_random_sigma_e(
    p,
    B,
    sparsity=0.3,
    weight_range_diag=(0.7, 1.2),
    weight_range_offdiag=(-0.7, -0.4, 0.4, 0.7),
):
    """
    Generate a random noise covariance matrix Sigma_e.
    Off-diagonal entries are allowed only if there is no directed edge
    between the two variables.
    """
    Sigma_e = np.zeros((p, p), dtype=np.float32)

    diag_values = np.random.uniform(weight_range_diag[0], weight_range_diag[1], p)
    np.fill_diagonal(Sigma_e, diag_values)

    for i in range(p):
        for j in range(i + 1, p):
            if B[i, j] == 0 and B[j, i] == 0:
                if np.random.rand() < sparsity:
                    weight = (
                        np.random.uniform(weight_range_offdiag[2], weight_range_offdiag[3])
                        if np.random.rand() > 0.5
                        else np.random.uniform(weight_range_offdiag[0], weight_range_offdiag[1])
                    )
                    Sigma_e[i, j] = weight
                    Sigma_e[j, i] = weight

    return Sigma_e

def generate_b_from_edges(p, edges):
    """
    Generate a weighted adjacency matrix B from a list of directed edges.

    Args:
        p: Number of variables.
        edges: List of tuples, e.g. [(0, 2), (1, 3)] means 0 -> 2 and 1 -> 3.
    """
    B = np.zeros((p, p), dtype=np.float32)

    for i, j in edges:
        if 0 <= i < p and 0 <= j < p and i != j:
            weight = (
                np.random.uniform(0.5, 2.0)
                if np.random.rand() > 0.5
                else np.random.uniform(-2.0, -0.5)
            )
            B[i, j] = weight
        else:
            print(f"Warning: Invalid edge ({i}, {j}) for p={p}. Skipping.")

    return B

def generate_sigma_from_biedges(
    p,
    bidirected_edges,
    weight_range_diag=(0.7, 1.2),
    weight_range_offdiag=(-0.7, -0.4, 0.4, 0.7),
):
    """
    Generate a symmetric Sigma_e matrix from bidirected edges.
    """
    Sigma_e = np.zeros((p, p), dtype=np.float32)

    diag_values = np.random.uniform(weight_range_diag[0], weight_range_diag[1], p)
    np.fill_diagonal(Sigma_e, diag_values)

    for i, j in bidirected_edges:
        if 0 <= i < p and 0 <= j < p and i != j:
            weight = (
                np.random.uniform(weight_range_offdiag[2], weight_range_offdiag[3])
                if np.random.rand() > 0.5
                else np.random.uniform(weight_range_offdiag[0], weight_range_offdiag[1])
            )
            Sigma_e[i, j] = weight
            Sigma_e[j, i] = weight
        else:
            print(f"Warning: Invalid bidirected edge ({i}, {j}) for p={p}. Skipping.")

    return Sigma_e

def make_positive_definite(Sigma, eps=1e-4):
    """
    Shift Sigma to make it positive definite if needed.
    """
    try:
        np.linalg.cholesky(Sigma)
        return Sigma
    except np.linalg.LinAlgError:
        min_eig = np.min(np.linalg.eigvals(Sigma).real)
        if min_eig < 0:
            Sigma = Sigma + (-min_eig + eps) * np.eye(Sigma.shape[0])

    return Sigma

def simulate_linear_sem(B, Sigma_e, n_samples):
    """
    Generate data from linear SEM:

        X = E (I - B)^(-1),
        E ~ N(0, Sigma_e)
    """
    p = B.shape[0]

    Sigma_e = make_positive_definite(Sigma_e)
    L = np.linalg.cholesky(Sigma_e)

    Z = np.random.randn(n_samples, p).astype(np.float32)
    E = Z @ L.T

    I_minus_B_inv = np.linalg.inv(np.eye(p) - B)
    X = E @ I_minus_B_inv

    return X.astype(np.float32)