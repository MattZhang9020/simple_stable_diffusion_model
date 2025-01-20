import utils
import model_loader

import torch

import numpy as np

from tokenizer import Tokenizer
from sampler import KEulerSampler

from PIL import Image
from tqdm import tqdm


def generate(prompts, uncond_prompts=None, input_images=None, cfg_scale=7.5, height=512, width=512, n_inference_steps=50, models={}, seed=None, device=None, idle_device=None):
    with torch.no_grad():
        if not isinstance(prompts, (list, tuple)) or not prompts:
            raise ValueError("prompts must be a non-empty list or tuple")
        
        if uncond_prompts and not isinstance(uncond_prompts, (list, tuple)):
            raise ValueError("uncond_prompts must be a non-empty list or tuple if provided")
        
        if uncond_prompts and len(prompts) != len(uncond_prompts):
            raise ValueError("length of uncond_prompts must be same as length of prompts")
        
        if input_images and len(prompts) != len(input_images):
            raise ValueError("length of input_images must be same as length of prompts")

        if height % 8 or width % 8:
            raise ValueError("height and width must be a multiple of 8")

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
            
        uncond_prompts = uncond_prompts or [""] * len(prompts)

        generator = torch.Generator(device=device)
        
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        tokenizer = Tokenizer()
        
        clip = models.get('clip')
        clip.to(device)

        cond_tokens = tokenizer.encode_batch(prompts)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        cond_context = clip(cond_tokens)
        
        uncond_tokens = tokenizer.encode_batch(uncond_prompts)
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)
        
        context = torch.cat([cond_context, uncond_context])
        
        # use the dtype of the model weights as our dtype
        dtype = clip.embedding.position_value.dtype
            
        to_idle(clip)
        del tokenizer, clip

        sampler = KEulerSampler(inference_steps=n_inference_steps, cfg_scale=cfg_scale)

        noise_shape = (len(prompts), 4, height // 8, width // 8)

        if input_images:
            encoder = models.get('encoder') or model_loader.load_encoder(device)
            encoder.to(device)
            
            processed_input_images = []
            for input_image in input_images:
                if type(input_image) is str:
                    input_image = Image.open(input_image)

                input_image = input_image.resize((width, height))
                input_image = np.array(input_image)
                input_image = torch.tensor(input_image, dtype=dtype)
                input_image = utils.rescale(input_image, (0, 255), (-1, 1))
                
                processed_input_images.append(input_image)
                
            input_images_tensor = torch.stack(processed_input_images).to(device)
            input_images_tensor = utils.move_channel(input_images_tensor, to="first")

            _, _, height, width = input_images_tensor.shape
            noise_shape = (len(prompts), 4, height // 8, width // 8)

            encoder_noise = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            latents = encoder(input_images_tensor, encoder_noise)

            latents_noise = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            latents += latents_noise * sampler.initial_scale

            to_idle(encoder)
            del encoder, processed_input_images, input_images_tensor, latents_noise
        else:
            latents = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            latents *= sampler.initial_scale

        diffusion = models.get('diffusion')
        diffusion.to(device)
        
        decoder = models.get('decoder')
        decoder.to(device)

        timesteps = tqdm(sampler.timesteps)
        for step_cnt, timestep in enumerate(timesteps):            
            latents = sampler.step(diffusion, latents, context, step_cnt, timestep)

        to_idle(diffusion)
        del diffusion
        
        images = decoder(latents)

        to_idle(decoder)
        del decoder

        images = utils.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = utils.move_channel(images, to="last")
        images = images.to('cpu', torch.uint8).numpy()

        return [Image.fromarray(image) for image in images]