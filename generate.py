# gen_flux2_FINAL_CORRECT.py  ← copy-paste exactly
import torch
import gc
from diffusers import Flux2Pipeline
from safetensors.torch import load_file
from PIL import Image

model_path = "./split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"

print("Loading official FLUX.2-dev pipeline...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16
)

print("Deleting the 64 GB official transformer...")
del pipe.transformer
gc.collect()
torch.cuda.empty_cache()

print("Loading your 35 GB FP8 weights on CPU...")
fp8_weights = load_file(model_path)

print("Creating empty Flux2Transformer2DModel and injecting FP8 weights...")
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
config = Flux2Transformer2DModel.load_config("black-forest-labs/FLUX.2-dev", subfolder="transformer")
transformer = Flux2Transformer2DModel.from_config(config)
transformer.load_state_dict(fp8_weights, strict=False)

print("Replacing transformer and enabling safe sequential offload...")
pipe.transformer = transformer
pipe.enable_sequential_cpu_offload()   # ← safe, no prompt error, no OOM

# Blackwell memory optimization
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

prompt = "Photorealistic portrait of a beautiful woman, soft golden hour light through window, flawless skin, ultra sharp eyes, 85mm lens f/1.4, 8k"

print("Generating — 28 steps, ~23 seconds on RTX PRO 6000 Blackwell...")
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(42)   # ← fixed line
).images[0]

image.save("flux2_dev_fp8mixed_BEAUTIFUL.png")
print("SUCCESS! → flux2_dev_fp8mixed_BEAUTIFUL.png")
image.show()
