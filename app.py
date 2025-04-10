import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "stabilityai/sdxl-turbo"

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = DiffusionPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype).to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    return image, seed

examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

with gr.Blocks() as demo:
    gr.Markdown(" # Text-to-Image Gradio Template")

    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt")
    run_button = gr.Button("Run")

    result = gr.Image(label="Result")
    seed_out = gr.Number(label="Used seed")

    with gr.Accordion("Advanced Settings", open=False):
        negative_prompt = gr.Textbox(label="Negative prompt", placeholder="(optional)", visible=True)
        seed = gr.Slider(0, MAX_SEED, step=1, value=0, label="Seed")
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        width = gr.Slider(256, MAX_IMAGE_SIZE, step=32, value=512, label="Width")
        height = gr.Slider(256, MAX_IMAGE_SIZE, step=32, value=512, label="Height")
        guidance_scale = gr.Slider(0, 10, step=0.1, value=0.0, label="Guidance scale")
        num_inference_steps = gr.Slider(1, 50, step=1, value=2, label="Steps")

    run_button.click(
        fn=infer,
        inputs=[prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result, seed_out],
    )

    gr.Examples(examples=examples, inputs=[prompt])

# ✅ 重點：Hugging Face Spaces 會抓 demo 這個 Blocks 物件為主 API
demo.launch()
