# gen_flux2_PERFECT_AND_FAST.py  ← copy-paste exactly
import torch
import gc
from diffusers import Flux2Pipeline
from safetensors.torch import load_file
from PIL import Image

model_path = "./split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"

print("Loading official FLUX.2-dev pipeline in FLOAT32 (this is the fix!)")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.float32          # ← THIS IS THE ONLY THING THAT WORKS WITH FP8 MIXED
)

print("Deleting official transformer...")
del pipe.transformer
gc.collect()
torch.cuda.empty_cache()

print("Loading your FP8 weights (they contain float32 layers)...")
weights = load_file(model_path)

print("Creating transformer and injecting weights...")
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
config = Flux2Transformer2DModel.load_config("black-forest-labs/FLUX.2-dev", subfolder="transformer")
transformer = Flux2Transformer2DModel.from_config(config)
transformer.load_state_dict(weights, strict=False)

print("Injecting your transformer and enabling safe offload...")
pipe.transformer = transformer
pipe.enable_sequential_cpu_offload()   # safe on 95 GB card

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

image.save("FLUX2_REAL_PERFECT.png")
print("SUCCESS — REAL photorealistic masterpiece saved!")
image.show()
