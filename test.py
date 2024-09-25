import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# Load local image
img_path = 'whale.jpg'  # Replace with the actual path to your local image
raw_image = Image.open(img_path).convert('RGB')

# conditional image captioning
text = "This is an image of "
inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
# inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))
# >>> a woman sitting on the beach with her dog
