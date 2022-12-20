import torch
import hashlib
import requests
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionImg2ImgPipeline
from io import BytesIO
from pathlib import Path
from PIL import Image

# params
model_path = "/home/prem/dev/models/prem-512-v0-prompts/3000"
# model_path = "runwayml/stable-diffusion-v1-5"
# model_path = "stabilityai/stable-diffusion-2-1-base"
images_dir = Path("/home/prem/dev/imgs_out/sg/prem-512-v0-3000-prompts_0_10")
num_images_per_prompt = 4
guidance_scale = 8.5 
num_inference_steps = 80
height = 512
width = 512

# prompts
negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
prompts = []
# prompts.append("a headshot of pknpknpknat man, 80mm lens")
# prompts.append("a digital painting of pknpknpknat man in a tuxedo, by Max Dauthendey, actor, close-up!!!!!, cute portrait, confident looking, oscar winning, symmetric portrait, hot")
# prompts.append("portrait of pknpknpknat man, D&D, muscular, robes, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha")
# prompts.append("pknpknpknat man boisterously dancing around the room by himself holding an empty wine bottle")
# prompts.append("ultrarealistic portrait of pknpknpknat man, cinematic lighting, award winning photo, 80mm lens")
# prompts.append("a photoshoot of pknpknpknat man, muscular, bodybuilder, 55mm lens")
# prompts.append("A profile picture of pknpknpknat man as an astonaut walking on the moon")
# prompts.append("An epic anime action picture of pknpknpknat man, black and white")
# prompts.append("An epic anime action picture of pknpknpknat man, colorful")

# setup
# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float32).to("cuda")

# Img to image
url = "https://pyxis.nymag.com/v1/imgs/128/67e/c245b5400d4a1895d7630d0af6542e9c2e-09-captain-america.rsquare.w330.jpg"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((512, 512))
# init_image = Image.open(r"/home/prem/dev/data/prem_512_selected/0013.jpg")
prompt = "A photo of pknpknpknat man"
images = pipe(prompt=prompt, init_image=init_image, num_inference_steps=200, strength=0.25, guidance_scale=7.5).images
images[0].save("im2im_test.png")

# if not images_dir.exists():
#     images_dir.mkdir(parents=True)

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
# for prompt_num, prompt in enumerate(prompts):
#     gen_prompt_images(prompt_num, prompt)
