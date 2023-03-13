#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from aitemplate.testing.benchmark_pt import benchmark_torch_function
import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
from aitemplate.utils.import_path import import_parent

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_controlnet_ait import \
    StableDiffusionControlnetAITPipeline

# Let's load the popular vermeer image
image = load_image(
    "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/test_imgs/bird.png"
)

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
controlnet.enable_xformers_memory_efficient_attention()
pipe = StableDiffusionControlnetAITPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, torch_device="cuda"
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(0)
# import ipdb;ipdb.set_trace()
with torch.autocast("cuda"):
    out_image = pipe(
        "bird", num_inference_steps=20, generator=generator, image=canny_image, torch_device="cuda"
    ).images[0]
    t = benchmark_torch_function(10, pipe, "bird", num_inference_steps=20, generator=generator, image=canny_image, torch_device="cuda")
    print(f"sd e2e: {t} ms")

out_image.save("controlnet_bird.png")
