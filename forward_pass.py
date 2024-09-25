# decoder_input_ids: tensor([[2]], device='cuda:0')

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

# generated_ids = model.generate(
#     input_ids=inputs["input_ids"],
#     pixel_values=inputs["pixel_values"],
#     max_new_tokens=1024,
#     do_sample=False,
#     num_beams=1
# )

# assert 5==6+1

inputs_embeds = model.get_input_embeds(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"])
attention_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], device=inputs_embeds.device).long()

# output_embeds = model.get_encoder().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

scripted_module = torch.jit.script(model.get_encoder())

print(scripted_module.code)

# scripted_encoder_model = torch.jit.script(model.get_encoder().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask))

# torch.jit.save(scripted_encoder_model, "scripted_encoder_model.pt")

# scripted_encoder_model.save("scripted_encoder_model.pt")
# print(type(scripted_encoder_model))
# if isinstance(scripted_encoder_model, dict):
#     torch.jit.save(torch.jit.script(torch.nn.ModuleDict(scripted_encoder_model)), "scripted_encoder_model.pt")

# torch.save(scripted_encoder_model,"scripted_encoder_model.pt")

# decoder_input_ids = torch.tensor(2, device='cuda:0').long().reshape(1,1)
# previous_past_keys = [[] for _ in range(6)]
# past_key_values = None
# encoder_outputs = output_embeds
# model.eval()

# decoder_input_ids = torch.tensor(0, device='cuda:0').long().reshape(1,1)


# past_key_values = torch.load("past_key_values_0.pt")
# attention_mask = torch.load("attention_mask_0.pt")
# inputs_embeds = torch.load("inputs_embeds_0.pt")

# dic = torch.load("whole_dict.pt")

# top_logit = model.language_model.forward(**dic)
# print(torch.argmax(top_logit.logits, dim=-1))
# print(top_logit.logits.reshape(-1)[:20])

# assert 5==6+1, "END!!"

# top_logit = model.language_model.forward(
#         inputs_embeds=inputs_embeds,
#         attention_mask=attention_mask,
#         input_ids=None,
#         decoder_input_ids=decoder_input_ids,
#         decoder_attention_mask=None,
#         encoder_outputs=encoder_outputs,
#         past_key_values = past_key_values,
#         use_cache=True
#     )

# print(torch.argmax(top_logit.logits, dim=-1))
# print(top_logit.logits.reshape(-1)[:20])

# assert 5==6+1

# arr = []
# for i in range(20):

#     # attention_mask = None
#     input_ids = None
    
#     decoder_attention_mask = None
    
#     use_cache = True
#     # print("start of inputs_embeds: ",inputs_embeds.shape)
#     print("decoder input is:", decoder_input_ids)
    

    
#     top_logit = model.language_model.forward(
        
#         attention_mask=attention_mask,
#         input_ids=input_ids,
#         decoder_input_ids=decoder_input_ids,
#         decoder_attention_mask=decoder_attention_mask,
#         encoder_outputs=encoder_outputs,
#         past_key_values = past_key_values,
#         use_cache=use_cache
#     )

#     past_key_values = top_logit.past_key_values

#     # print("shape start: ",past_key_values[0][0].shape)

#     # past_key_values_new = ()
#     # for i in range(6):
#     #     past_key_values_new += ((past_key_values[i][0], past_key_values[i][1]),)
        
#     log = top_logit.logits
    
#     # probs = torch.softmax(log[0, -1], dim=-1)
#     # next_tokens = torch.multinomial(probs, num_samples=1)
    
#     next_tokens = torch.argmax(log, dim=-1)
#     decoder_input_ids = next_tokens
#     log = log.reshape(-1)
#     print(log[:20])
    
#     arr.append(next_tokens)

# print(arr)
            





#! Returned
#! tuple of lenght 6 -> inside each tuple we have a tuple of length 4 -> each of this is a tensor of shape [1, 12, N, 64]

#! Passed
#! Tuple of Length 6 -> Each having a tuple of length 2 -> tensor of shape [1,12,N,64]