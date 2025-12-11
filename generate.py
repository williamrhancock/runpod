import torch
import gc
import os
from diffusers import Flux2Pipeline
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from safetensors.torch import load_file
from PIL import Image

# Your model path (update if using Kijai's)
model_path = "./split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"

# Auto-redownload if corrupted (run once)
if not os.path.exists(model_path) or os.path.getsize(model_path) < 35 * 1024**3:  # <35 GB = corrupted
    print("Detected corrupted/incomplete file. Redownloading...")
    os.system(f"hf download Comfy-Org/flux2-dev split_files/diffusion_models/flux2_dev_fp8mixed.safetensors --local-dir ./ --force-download")
    print("Redownload complete. Rerun script if needed.")

print("1. Loading official FLUX.2-dev pipeline...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.float32  # Matches FP8 mixed (avoids dtype mismatch)
)

print("2. Deleting official transformer...")
del pipe.transformer
gc.collect()
torch.cuda.empty_cache()

print("3. Loading FP8 weights...")
fp8_weights = load_file(model_path)

print("4. Creating transformer and injecting weights...")
config = Flux2Transformer2DModel.load_config("black-forest-labs/FLUX.2-dev", subfolder="transformer")
transformer = Flux2Transformer2DModel.from_config(config)
transformer.load_state_dict(fp8_weights, strict=False)

print("5. Injecting transformer...")
pipe.transformer = transformer
pipe.enable_sequential_cpu_offload()  # Safe for 95 GB VRAM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

prompt = "Photorealistic portrait of a young woman with flawless skin, soft golden hour light through window, ultra sharp details, 85mm lens f1.4, 8k"

print("6. Generating (28 steps, ~23s on RTX PRO 6000 Blackwell)...")
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save("flux2_fixed.png")
print("SUCCESS! → flux2_fixed.png")
image.show()
