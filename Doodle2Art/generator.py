# generator.py
import torch
import gc
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from PIL import Image

DEVICE = "cpu"


def generate_art(sketch_path, prompt, output_path="final_art.png"):
    """
    CPU-only Stable Diffusion + ControlNet watercolor generator.
    """

    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # Load Stable Diffusion pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None,
        low_cpu_mem_usage=True
    )

    # Faster scheduler for CPU
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    # Reduce RAM usage
    pipe.enable_attention_slicing()
    pipe.to(DEVICE)

    sketch = Image.open(sketch_path).convert("RGB")

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            image=sketch,
            num_inference_steps=15,   # CPU sweet spot
            guidance_scale=6.5
        ).images[0]

    image.save(output_path)

    # Cleanup
    del pipe, controlnet
    gc.collect()

    return output_path


if __name__ == "__main__":
    generate_art(
        "processed_doodle.png",
        "a watercolor painting of the drawn subject, clean outline"
    )