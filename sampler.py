import torch

import numpy as np

class KEulerSampler():
    def __init__(self, beta_start=0.00085, beta_end=0.012, inference_steps=50, training_steps=1000, cfg_scale=7.5):
        self.timesteps = np.linspace(training_steps - 1, 0, inference_steps)

        betas = self._sigmoid_beta_schedule(training_steps, beta_start, beta_end).cpu().numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(self.timesteps, range(training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)
        
        self.sigmas = sigmas
        
        self.initial_scale = sigmas.max()
        
        self.cfg_scale = cfg_scale
    
    def _sigmoid_beta_schedule(self, timesteps, beta_start, beta_end):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        
    # def _embed_sinusoidal_position(self, timestep, max_period=10000, time_embed_dim=320):
    #     half = time_embed_dim // 2
        
    #     log_max_period = np.log(max_period)
        
    #     arange_vector = np.arange(start=0, stop=half, dtype=np.float32)
    #     frac_vector = arange_vector / half
        
    #     log_scaled = -log_max_period * frac_vector
        
    #     freqs = np.exp(log_scaled)
        
    #     args = timestep[:, np.newaxis] * freqs[np.newaxis, :]
        
    #     embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        
    #     return embedding
    
    def _get_time_embedding(self, timestep, dtype):
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=dtype) / 160)
        x = torch.tensor([timestep], dtype=dtype)[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    
    def _get_input_scale(self, time_step):
        sigma = self.sigmas[time_step]
        return 1 / (sigma ** 2 + 1) ** 0.5
    
    def step(self, model, x_T, context, step_cnt, t):
        x_t = x_T * self._get_input_scale(step_cnt)
        x_t = x_t.repeat(2, 1, 1, 1)

        t_embed = self._get_time_embedding(t, dtype=x_t.dtype).to(x_t.device)
        # t_embed = torch.tensor(t_embed, device=x_t.device, dtype=torch.float)
        
        eps = model(x_t, context, t_embed)
        
        eps_cond, eps_uncond = eps.chunk(2)
        eps = self.cfg_scale * (eps_cond - eps_uncond) + eps_uncond
        
        sigma_from = self.sigmas[step_cnt]
        sigma_to = self.sigmas[step_cnt + 1]

        x_T = x_T + eps * (sigma_to - sigma_from)
        
        return x_T