
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .models import Generator, Discriminator
from .losses import acyclicity_loss, simplicity_loss
from .utils import (
    sample_weights_B,
    sample_weights_Sigma,
    get_prob_B,
    get_prob_Sigma,
    threshold_structure,
    get_true_structures,
)

def gan_dag(data, config, B_true=None, Sigma_e_true=None, device=None, verbose=True):
    """
    Estimate directed structure B and bidirected structure Sigma using Gumbel-GAN.

    Args:
        data: np.ndarray, shape [n_samples, p].
        config: Config object.
        B_true: optional true weighted B.
        Sigma_e_true: optional true Sigma.
        device: "cpu" or "cuda".
        verbose: whether to print training information.

    Returns:
        dict containing true structures, learned structures, probabilities and models.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    p = config.p

    X_tensor = torch.from_numpy(data.astype("float32")).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    generator = Generator(p).to(device)
    discriminator = Discriminator(p).to(device)

    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.AdamW(generator.parameters(), lr=config.lr_g)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=config.lr_d)

    current_tau = config.tau_start
    cholesky_fail_count = 0

    for epoch in range(config.num_epochs):
        for batch_idx, (real_data_batch,) in enumerate(dataloader):
            current_batch_size = real_data_batch.shape[0]

            weights_B = sample_weights_B(p, device=device)
            weights_Sigma = sample_weights_Sigma(p, device=device)

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            real_labels = torch.ones(current_batch_size, 1, device=device)
            Z_for_G = torch.randn(current_batch_size, p, device=device)

            (
                gen_fake_data,
                B_for_G,
                Sigma_for_G,
                logit_Sigma_for_G,
                cholesky_success,
            ) = generator(
                Z_for_G,
                weights_B,
                weights_Sigma,
                current_tau,
            )

            if not cholesky_success:
                cholesky_fail_count += 1
                if verbose and cholesky_fail_count <= 5:
                    print(
                        f"Cholesky decomposition or inverse failed "
                        f"at epoch {epoch + 1}, batch {batch_idx + 1}."
                    )
                    print("Sigma logits:")
                    print(logit_Sigma_for_G)
                    print("Sigma:")
                    print(Sigma_for_G)

            disc_output_on_fake = discriminator(gen_fake_data)

            g_loss_adversarial = adversarial_loss(
                disc_output_on_fake,
                real_labels,
            )

            g_loss_acyclicity = acyclicity_loss(B_for_G)

            g_loss_simple = simplicity_loss(Sigma_for_G)

            g_loss = (
                g_loss_adversarial
                + config.lambda_acyc * g_loss_acyclicity
                + config.lambda_simple * g_loss_simple
            )

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_output = discriminator(real_data_batch)
            d_loss_real = adversarial_loss(real_output, real_labels)

            Z = torch.randn(current_batch_size, p, device=device)
            fake_data, _, _, _, _ = generator(
                Z,
                weights_B,
                weights_Sigma,
                current_tau,
            )

            fake_labels = torch.zeros(current_batch_size, 1, device=device)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            optimizer_D.step()

        current_tau = max(config.tau_end, current_tau * config.anneal_rate)

        if verbose and (epoch + 1) % 100 == 0:
            print(
                f"[Epoch {epoch + 1}/{config.num_epochs}] "
                f"[D loss: {d_loss.item():.4f}] "
                f"[G loss: {g_loss.item():.4f}] "
                f"[Temp tau: {current_tau:.4f}]"
            )

        if verbose and (epoch + 1) % 1000 == 0:
            with torch.no_grad():
                current_prob_B = get_prob_B(generator)
                current_prob_Sigma = get_prob_Sigma(generator)

                print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
                print("Current prob_B:")
                print(current_prob_B)
                print("Current prob_Sigma:")
                print(current_prob_Sigma)

    # ---------------------
    # Extract results
    # ---------------------
    with torch.no_grad():
        prob_B = get_prob_B(generator).detach().cpu()
        prob_Sigma = get_prob_Sigma(generator).detach().cpu()

        B_learned_structure = threshold_structure(prob_B, ratio=0.7)
        Sigma_learned_structure = threshold_structure(prob_Sigma, ratio=0.7)

    if B_true is not None and Sigma_e_true is not None:
        B_true_structure, Sigma_true_structure = get_true_structures(
            B_true,
            Sigma_e_true,
        )
    else:
        B_true_structure = None
        Sigma_true_structure = None

    return {
        "B_true_structure": B_true_structure,
        "Sigma_true_structure": Sigma_true_structure,
        "B_learned_structure": B_learned_structure,
        "Sigma_learned_structure": Sigma_learned_structure,
        "prob_B": prob_B,
        "prob_Sigma": prob_Sigma,
        "generator": generator,
        "discriminator": discriminator,
    }