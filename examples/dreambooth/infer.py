import torch
import hashlib
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pathlib import Path

# params
model_path = "/home/prem/dev/models/sd-2-1-sg_v0/2000"
images_dir = Path("/home/prem/dev/imgs_out/sg/sd-2-1-sg_v0")
num_images_per_prompt = 4
guidance_scale = 8.5 
num_inference_steps = 80
height = 512
width = 512

# prompts
negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
prompts = []
prompts.append("a headshot of sgsgsg man, 80mm lens")
prompts.append("a digital painting of sgsgsg man in a tuxedo, by Max Dauthendey, actor, close-up!!!!!, cute portrait, confident looking, oscar winning, symmetric portrait, hot")
prompts.append("portrait of sgsgsg man, D&D, muscular, robes, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha")
prompts.append("sgsgsg man boisterously dancing around the room by himself holding an empty wine bottle")
prompts.append("ultrarealistic portrait of sgsgsg man, cinematic lighting, award winning photo, 80mm lens")

# setup
# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float32).to("cuda")

if not images_dir.exists():
    images_dir.mkdir(parents=True)

# inference function per prompt
def gen_prompt_images(num, prompt):
    images = pipe(prompt, height=height, width=width, negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale).images
    for image in images:
        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
        image_filename = images_dir / f"prompt-{num}-{hash_image}.jpg"
        image.save(image_filename)

# generate images
for prompt_num, prompt in enumerate(prompts):
    gen_prompt_images(prompt_num, prompt)
