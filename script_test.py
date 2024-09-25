from modeling_florence2 import Florence2ForConditionalGeneration
from transformers import AutoProcessor
import torch
from PIL import Image

model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True,torch_dtype=torch.float16)

prompt = "<MORE_DETAILED_CAPTION>"
device = "cuda:0"

model = model.to(device)

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

img_path = 'whale.jpg'  # Replace with the actual path to your local image
image = Image.open(img_path).convert('RGB')
torch_dtype = torch.float16

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1
)

print(generated_ids)

# inputs_embeds = model.get_input_embeds(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"])

# attention_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], device=inputs_embeds.device).long()

# top_logit = model.forward(input_ids=inputs_embeds, attention_mask=attention_mask)
# print(top_logit.shape)


# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
# parsed_answer = processor.post_process_generation(generated_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height))

# print(parsed_answer)


