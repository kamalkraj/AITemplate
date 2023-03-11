from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image

# Let's load the popular vermeer image
image = load_image(
    "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/test_imgs/bird.png"
)
from aitemplate.testing.benchmark_pt import benchmark_torch_function
import cv2
from PIL import Image
import numpy as np

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# # this command loads the individual model components on GPU on-demand.
# pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)
# import ipdb; ipdb.set_trace()
# print(canny_image.shape)
pipe.enable_xformers_memory_efficient_attention()

out_image = pipe(
    "bird", num_inference_steps=20, generator=generator, image=canny_image
).images[0]

t = benchmark_torch_function(10, pipe, "bird", num_inference_steps=20, generator=generator, image=canny_image)
print(f"sd e2e: {t} ms")

out_image.save("controlnet_pt_bird.png")