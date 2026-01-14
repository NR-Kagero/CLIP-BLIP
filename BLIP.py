import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = AutoModelForVisualQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base",
    dtype=torch.float16,
    device_map="auto"
)

image = Image.open(r".\download.jpg")

question = "What is the weather in this image?"
inputs = processor(images=image, text=question, return_tensors="pt").to(model.device, torch.float16)

output = model.generate(**inputs)
print(processor.batch_decode(output, skip_special_tokens=True)[0])

