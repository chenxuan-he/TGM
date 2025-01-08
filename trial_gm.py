import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from functions import permutation_test

# ------------------------
# GAN Model
# ------------------------

class GAN:
    def __init__(self, data_dim, noise_dim):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, data_dim)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def train(self, real_data, epochs=1000, batch_size=64, lr=0.001):
        real_data = torch.tensor(real_data, dtype=torch.float32)
        dataloader = DataLoader(real_data, batch_size=batch_size, shuffle=True)
        g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            for real_samples in dataloader:
                # Train Discriminator
                d_optimizer.zero_grad()
                noise = torch.randn(len(real_samples), self.noise_dim)
                fake_samples = self.generator(noise)
                real_preds = self.discriminator(real_samples)
                fake_preds = self.discriminator(fake_samples.detach())
                d_loss = loss_fn(real_preds, torch.ones_like(real_preds)) + \
                         loss_fn(fake_preds, torch.zeros_like(fake_preds))
                d_loss.backward()
                d_optimizer.step()
                # Train Generator
                g_optimizer.zero_grad()
                noise = torch.randn(len(real_samples), self.noise_dim)
                fake_samples = self.generator(noise)
                fake_preds = self.discriminator(fake_samples)
                g_loss = loss_fn(fake_preds, torch.ones_like(fake_preds))
                g_loss.backward()
                g_optimizer.step()
    def sample(self, n):
        with torch.no_grad():
            noise = torch.randn(n, self.noise_dim)
            return self.generator(noise).numpy()

# ------------------------
# Diffusion Model
# ------------------------

class SimpleDiffusionModel(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, data_dim)
        )
    def forward(self, x, t):
        """
        Forward pass with input `x` and timestep `t`.
        """
        t_embed = t.unsqueeze(-1)  # Ensure t_embed has shape (batch_size, 1)
        # t_embed = t_embed.expand(-1, x.shape[-1])  # Broadcast `t` to match `x`
        return self.net(torch.cat([x, t_embed], dim=1))  # Concatenate along the feature dimension
    

def train_diffusion_model(data, n_steps=100, epochs=500, batch_size=64, lr=1e-3):
    """
    Train a simple DDPM model.
    """
    data_dim = data.shape[1]
    model = SimpleDiffusionModel(data_dim).to(torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(torch.tensor(data, dtype=torch.float32), batch_size=batch_size, shuffle=True)
    # Define noise schedule
    beta = torch.linspace(0.0001, 0.02, n_steps)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    # begin training
    for _ in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            batch_size = batch.size(0)
            t = torch.randint(0, n_steps, (batch_size,))
            alpha_t = alpha_cumprod[t].unsqueeze(-1)
            noise = torch.randn_like(batch)
            noisy_data = torch.sqrt(alpha_t) * batch + torch.sqrt(1 - alpha_t) * noise
            pred_noise = model(noisy_data, t.float())
            loss = nn.MSELoss()(pred_noise, noise)
            loss.backward()
            optimizer.step()
    def sample(n_samples):
        """
        Generate samples using the diffusion process.
        """
        samples = torch.randn((n_samples, data_dim))
        for t in reversed(range(n_steps)):
            alpha_t = alpha_cumprod[t]
            beta_t = beta[t]
            noise = torch.randn_like(samples) if t > 0 else 0
            t_tensor = torch.full((n_samples,), t, dtype=torch.float32)
            samples = (samples - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod[t]) * model(samples, t_tensor)) / torch.sqrt(alpha_t)
            samples += torch.sqrt(beta_t) * noise
        return samples.detach().numpy()
    return sample


# ------------------------
# Flow Model
# ------------------------

class RealNVP(nn.Module):
    def __init__(self, data_dim, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.data_dim = data_dim
        self.masks = [torch.arange(data_dim) % 2 for _ in range(num_blocks)]
        self.s = nn.ModuleList([nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh()
        ) for _ in range(num_blocks)])
        self.t = nn.ModuleList([nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        ) for _ in range(num_blocks)])
    def forward(self, x, reverse=False):
        """
        Forward pass of the normalizing flow.
        """
        log_det_jacobian = 0
        if reverse:
            for mask, s, t in zip(reversed(self.masks), reversed(self.s), reversed(self.t)):
                masked_x = x * mask
                scale = torch.exp(-s(masked_x))
                x = masked_x + scale * (x - t(masked_x))
                log_det_jacobian -= torch.sum(-s(masked_x), dim=-1)
        else:
            for mask, s, t in zip(self.masks, self.s, self.t):
                masked_x = x * mask
                scale = torch.exp(s(masked_x))
                x = masked_x + scale * x + t(masked_x)
                log_det_jacobian += torch.sum(s(masked_x), dim=-1)
        return x, log_det_jacobian
    # sample
    def sample(self, n_samples):
        """
        Generate samples by sampling from a base distribution.
        """
        z = torch.randn((n_samples, self.data_dim))
        samples, _ = self.forward(z, reverse=True)
        return samples.detach().numpy()

def train_flow_model(data, epochs=500, batch_size=64, lr=1e-3):
    """
    Train a RealNVP model.
    """
    data_dim = data.shape[1]
    model = RealNVP(data_dim).to(torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(torch.tensor(data, dtype=torch.float32), batch_size=batch_size, shuffle=True)
    # begin training
    for _ in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            z, log_det_jacobian = model(batch)
            base_log_prob = -0.5 * torch.sum(z ** 2, dim=-1) - 0.5 * data_dim * np.log(2 * np.pi)
            loss = -(base_log_prob + log_det_jacobian).mean()
            loss.backward()
            optimizer.step()
    return model.sample



# ------------------------
# Main Execution
# ------------------------

from data_generation import generate_standard_normal

n = 200  # Number of original samples
p = 4    # Dimension of the data
n_gen_samples = 200  # Number of generated samples
nsim = 100  # Number of permutations
n = 200
X = generate_standard_normal(n, p)

# Train GAN
gan = GAN(data_dim=p, noise_dim=p)
gan.train(X, epochs=500)
gan_samples = gan.sample(n_gen_samples)

# Train Diffusion Model
# diffusion_model = train_diffusion_model(X)
# diffusion_samples = diffusion_model(n_gen_samples)

# Train Flow Model
# flow_model = train_flow_model(X)
# flow_samples = flow_model(n_gen_samples)

from functions import RectifiedFlow, train_rectified_flow, MLP
gaussian_variable = torch.randn(n, p, dtype=torch.float32)
iterations = 100
batchsize = 500
x1_pairs = torch.stack([gaussian_variable, torch.tensor(X, dtype=torch.float32)], dim=1)
rectified_flow_1 = RectifiedFlow(model=MLP(input_dim=p+1, output_dim=p, hidden_num=256), num_steps=100)
optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr=5e-3)
rectified_flow_1, loss_curve1 = train_rectified_flow(rectified_flow_1, optimizer, x1_pairs, batchsize, iterations)

gaussian_variable2 = torch.randn(n, p)
flow_samples = np.array(rectified_flow_1.sample_ode(gaussian_variable2)[-1])

# Perform the permutation test

permutation_test(X, gan_samples, nsim=100)
permutation_test(X, np.array(flow_samples), nsim=100)

