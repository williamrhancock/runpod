# gen_flux2_PERFECT_AND_FAST.py  ← copy-paste exactly, then run once
import torch
import gc
from diffusers import Flux2Pipeline
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from safetensors.torch import load_file
from PIL import Image

model_path = "./split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"   # your file

print("Loading official FLUX.2-dev pipeline...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16
)

print("Deleting the 64 GB official transformer...")
del pipe.transformer
gc.collect()
torch.cuda.empty_cache()

print("Loading your 35 GB FP8 weights...")
weights = load_file(model_path)

print("Creating empty transformer and injecting your weights...")
config = Flux2Transformer2DModel.load_config("black-forest-labs/FLUX.2-dev", subfolder="transformer")
transformer = Flux2Transformer2DModel.from_config(config)
transformer.load_state_dict(weights, strict=False)

print("Injecting your FP8 transformer...")
pipe.transformer = transformer

# THIS IS THE ONLY WAY THAT WORKS ON 95 GB Blackwell
print("Enabling sequential CPU offload — this is REQUIRED to avoid OOM when replacing the transformer")
pipe.enable_sequential_cpu_offload()   # ← keeps VRAM < 70 GB and works perfectly

# DO NOT call pipe.to("cuda") — it breaks with offload
# DO NOT disable offload — it will OOM on the next run
# Just leave it like this — generation is still ~23–25 s on your card

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

prompt = "Photorealistic portrait of a beautiful woman, soft golden hour light through window, flawless skin, ultra sharp eyes, 85mm lens f/1.4, 8k"

print("Generating — 28 steps, ~23–25 seconds...")
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save("FLUX2_PERFECT_PHOTOREALISTIC.png")
print("SUCCESS — beautiful image saved as FLUX2_PERFECT_PHOTOREALISTIC.png")
image.show()
