import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, Florence2ForConditionalGeneration
from transformers import BitsAndBytesConfig, AutoTokenizer


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def inspect_tokenizer(tokenizer):
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    
    # Check if it's a fast tokenizer
    print(f"Is fast tokenizer: {tokenizer.is_fast}")
    
    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    print("First 10 vocab items:", list(vocab.items())[:10])
    
    # Check for special tokens
    print(f"PAD token: {tokenizer.pad_token}")
    print(f"UNK token: {tokenizer.unk_token}")
    print(f"SEP token: {tokenizer.sep_token}")
    print(f"CLS token: {tokenizer.cls_token}")
    print(f"MASK token: {tokenizer.mask_token}")
    
    # Check tokenizer specific attributes
    if hasattr(tokenizer, 'do_lower_case'):
        print(f"Do lower case: {tokenizer.do_lower_case}")
    
    if hasattr(tokenizer, 'model_max_length'):
        print(f"Model max length: {tokenizer.model_max_length}")
    
    # For BPE tokenizers
    if hasattr(tokenizer, 'bpe_ranks'):
        print("This is a BPE-based tokenizer")
        print(f"Number of merges: {len(tokenizer.bpe_ranks)}")
    
    # For WordPiece tokenizers
    if hasattr(tokenizer, 'wordpiece_tokenizer'):
        print("This is a WordPiece tokenizer")
    
    # For SentencePiece tokenizers
    if hasattr(tokenizer, 'sp_model'):
        print("This is a SentencePiece tokenizer")
    
    # Test tokenization
    test_text = "Hello, how are you doing today?"
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokenized test text: {tokens}")
    
    # Get the backend tokenizer for fast tokenizers
    if tokenizer.is_fast:
        backend_tokenizer = tokenizer.backend_tokenizer
        print(f"Backend tokenizer: {type(backend_tokenizer).__name__}")

# Usage



quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base-ft",
    quantization_config=quantization_config,
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, torch_dtype=torch.float16)


tokenizer = processor.tokenizer
print(hasattr(tokenizer, 'is_fast') and tokenizer.is_fast)

# prompt = "<MORE_DETAILED_CAPTION>"

# img_path = 'whale.jpg'  # Replace with the actual path to your local image
# image = Image.open(img_path).convert('RGB')

# tokenizer = AutoTokenizer.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

# inspect_tokenizer(tokenizer)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

print(inputs['pixel_values'].shape)

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'inputs' is your tensor
tensor = inputs['pixel_values']

# # Convert to CPU, numpy, and transpose
# img_array = tensor.cpu().numpy().squeeze().transpose(1, 2, 0)
# img_array = np.array(img_array, dtype=np.float32)

# # Normalize if necessary (if values are not in [0, 1] range)
# img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

# # Display the image
# plt.imshow(img_array)
# plt.axis('off')
# plt.show()

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# # parsed_answer = processor.post_process_generation(generated_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height))

# # print(parsed_answer)
