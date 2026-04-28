
import numpy as np

from gumbel_gan_dag.config import Config
from gumbel_gan_dag.data import (
    generate_b_from_edges,
    generate_sigma_from_biedges,
    simulate_linear_sem,
)
from gumbel_gan_dag.train import gan_dag
from gumbel_gan_dag.utils import set_seed

def main():
    set_seed(42)

    config = Config()

    # Directed edges and bidirected edges
    edges = [(0, 2)]
    biedges = [(0, 1), (1, 2), (1, 3), (2, 3)]

    B_true = generate_b_from_edges(config.p, edges)
    Sigma_e_true = generate_sigma_from_biedges(config.p, biedges)

    print("True B:")
    print(np.round(B_true, 2))

    print("\nTrue Sigma_e:")
    print(np.round(Sigma_e_true, 2))

    X_data_np = simulate_linear_sem(
        B=B_true,
        Sigma_e=Sigma_e_true,
        n_samples=config.n_samples,
    )

    print("\nData generated successfully.")
    print("\nStarting GAN training...")

    results = gan_dag(
        data=X_data_np,
        config=config,
        B_true=B_true,
        Sigma_e_true=Sigma_e_true,
        verbose=True,
    )

    print("\nGAN training finished.")

    print("\n--- Estimation Result ---")

    print("\nTrue B structure:")
    print(results["B_true_structure"])

    print("\nLearned B structure:")
    print(results["B_learned_structure"])

    print("\nTrue Sigma structure:")
    print(results["Sigma_true_structure"])

    print("\nLearned Sigma structure:")
    print(results["Sigma_learned_structure"])

    print("\nLearned prob_B:")
    print(results["prob_B"])

    print("\nLearned prob_Sigma:")
    print(results["prob_Sigma"])

if __name__ == "__main__":
    main()