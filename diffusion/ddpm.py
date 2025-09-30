# diffusion/ddpm.py
import torch
import torch.nn.functional as F
import numpy as np


class DDPM:
    """
    Minimal DDPM utilities for training & sampling.
    Model is expected to predict epsilon (noise).

    ⚠️ Initialization Info:
    - This DDPM was initialized with a specific loss_type (default = "mse").
    - Loss options can be extended (mse, l1, huber, etc).
    """

    def __init__(self, timesteps=1000, beta_schedule="linear", loss_type="mse"):
        self.timesteps = timesteps
        self.loss_type = loss_type

        # define available loss functions
        self.loss_fns = {
            "mse": F.mse_loss,
            "l1": F.l1_loss,
            "huber": lambda input, target: F.smooth_l1_loss(input, target, beta=1.0),
        }
        if self.loss_type not in self.loss_fns:
            raise ValueError(f"[DDPM INIT ERROR] Unknown loss_type '{self.loss_type}'")

        # print banner
        print("=" * 80)
        print("🌀 DDPM Initialized")
        print(f" Timesteps    : {self.timesteps}")
        print(f" Beta schedule: {beta_schedule}")
        print(f" Loss type    : {self.loss_type.upper()}")
        print("=" * 80)

        self.set_beta_schedule(beta_schedule)

    def set_beta_schedule(self, name):
        if name == "linear":
            beta_start, beta_end = 1e-4, 2e-2
            self.betas = torch.linspace(beta_start, beta_end, self.timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {name}")

        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(1.0 - self.betas) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

    # ---------- q(x_t | x_0) ----------
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ab * x_start + sqrt_om * noise

    # ---------- p(x_{t-1} | x_t) ----------
    def p_mean_variance(self, model, x_t, t, cond):
        eps_pred = model(x_t, t, cond)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred).clamp(-1.0, 1.0)

        model_mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + \
                     self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        model_var = self.extract(self.posterior_variance, t, x_t.shape)
        model_log_var = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return x0_pred, model_mean, model_var, model_log_var

    def predict_x0_from_eps(self, x_t, t, eps):
        return self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def extract(self, arr, timesteps, x_shape):
        out = arr.gather(0, timesteps)
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out.view(-1, *([1] * (len(x_shape) - 1))).expand(x_shape)

    # ---------- training loss ----------
    def loss(self, model, x0, t, cond):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model(x_t, t, cond)
        return self.loss_fns[self.loss_type](eps_pred, noise)

    # ---------- sampling ----------
    @torch.no_grad()
    def sample(self, model, shape, cond, device, progress=False):
        img = torch.randn(shape, device=device)
        B = shape[0]
        rng = range(self.timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            rng = tqdm(rng)

        for t_idx in rng:
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            x0, mean, _, log_var = self.p_mean_variance(model, img, t, cond)
            if t_idx > 0:
                noise = torch.randn_like(img)
                img = mean + torch.exp(0.5 * log_var) * noise
            else:
                img = mean
        return img.clamp(-1, 1)
