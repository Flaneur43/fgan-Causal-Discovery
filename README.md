# fgan-Causal-Discovery
# Generative Causal Discovery with Unmeasured Confounding via Bayesian f-GAN

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./paper.pdf) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Note:** The full manuscript is currently under review. You can download the preprint [here](./fgancd.pdf).

## 📖 Abstract

Causal discovery from observational data is challenging, especially in the presence of **unmeasured confounding**. Traditional methods often rely on point estimation of parameters (e.g., BIC score), which can be problematic for singular models. 

This project proposes a novel **Bayesian framework** that:
1.  **Marginalizes over weights** using a stochastic generator, focusing purely on learning the binary structure ($S_B, S_\Sigma$).
2.  Minimizes the **Bayes Free Energy** via variational **f-divergence** minimization.
3.  Utilizes **Gumbel-Softmax** for differentiable discrete structure search within an adversarial (f-GAN) training loop.

## 🚀 Key Method

### 1. From Bayes Free Energy to f-GAN
We aim to minimize the Bayes Free Energy, which is equivalent to minimizing the KL divergence between the true data distribution and the model's marginal distribution. By leveraging the **f-GAN framework** [Nowozin et al., 2016], we transform this into a min-max optimization problem:

$$
\min_{G} \max_{D} \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{x \sim Q_G}[\log(1 - D(x))]
$$

### 2. Generative Process with Weight Marginalization
Unlike NOTEARS, our generator does not learn fixed weights. Instead, it learns the **probability of edges** (logits). During training, weights are sampled from a prior distribution $\phi(w)$ to simulate the marginal likelihood integral.

![Model Architecture](assets/architecture_diagram.png)
*(The generator outputs a structural mask using Gumbel-Softmax, combined with random weights to generate fake data.)*

## 🧪 Experiments (Proof of Concept)

We evaluated the method on synthetic datasets with linear SEMs and unmeasured confounding.

| Metric | Directed Edges ($S_B$) | Confounding ($S_\Sigma$) |
| :--- | :---: | :---: |
| **SHD** | Low | Low |
| **TPR** | High | High |

*See the [Full Paper](./paper.pdf) for detailed experimental setup and results.*

## 💻 Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/fgan-causal-discovery.git

# Install dependencies
pip install -r requirements.txt

# Run the toy example
python main.py
