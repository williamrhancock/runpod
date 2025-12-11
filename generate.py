import torch
import gc
from diffusers import Flux2Pipeline
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from safetensors.torch import load_file
from PIL import Image

model_path = "./split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"

print("1. Loading official FLUX.2-dev pipeline...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16
)

print("2. Deleting the 64 GB official transformer...")
del pipe.transformer
gc.collect()
torch.cuda.empty_cache()

print("3. Loading your FP8 weights on CPU...")
fp8_state_dict = load_file(model_path)
print(f"   → {len(fp8_state_dict)} tensors loaded")

print("4. Loading transformer config dict (tiny, cached)...")
transformer_config_dict = Flux2Transformer2DModel.load_config("black-forest-labs/FLUX.2-dev", subfolder="transformer")

print("5. Creating empty Flux2Transformer2DModel...")
transformer = Flux2Transformer2DModel.from_config(transformer_config_dict)

print("6. Injecting your FP8 weights...")
transformer.load_state_dict(fp8_state_dict, strict=False)

print("7. Moving transformer to GPU...")
pipe.transformer = transformer.to("cuda")

print("8. Moving the rest of the pipeline to GPU...")
pipe.to("cuda")

# Blackwell memory optimization
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

prompt = "Photorealistic portrait of a young woman with flawless skin, soft golden hour light through window, ultra sharp details, 85mm lens f1.4, 8k"

print("9. Generating (28 steps, ~25s on RTX PRO 6000 Blackwell)...")
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save("flux2_dev_fp8mixed_perfect.png")
print("SUCCESS! → flux2_dev_fp8mixed_perfect.png")
image.show()
