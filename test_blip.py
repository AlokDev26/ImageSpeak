
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Device select karo
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 2. Pretrained BLIP model load karo
model_name = "Salesforce/blip-image-captioning-base"

print("Loading BLIP model... First download mein time lag sakta hai.")

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)
model.eval()

print("BLIP model loaded successfully!")

# 3. Test image ka path
image_path = input(
    "\nImage ka complete path paste karo: "
).strip().strip('"')

image_file = Path(image_path)

if not image_file.is_file():
    raise FileNotFoundError(f"Image nahi mili: {image_file}")

# 4. Image open karo
image = Image.open(image_file).convert("RGB")

# 5. Caption generate karo
inputs = processor(images=image, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        num_beams=5
    )

caption = processor.decode(
    output[0],
    skip_special_tokens=True
)

print("\n" + "=" * 50)
print("BLIP GENERATED CAPTION:")
print(caption)
print("=" * 50)
