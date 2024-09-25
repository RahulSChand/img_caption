import torch
from modeling_florence2 import Florence2ForConditionalGeneration


def print_model_params():
    # Load the model state dict from the .bin file
    # state_dict = torch.load(file_path, map_location=torch.device('cpu'))

    model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True,torch_dtype=torch.float16)

    print(model.hf_cache_dir)

    # state_dict = model.state_dict()


    # # Iterate through all parameters and print their names and shapes
    # for param_name, param_tensor in state_dict.items():
    #     # print(f"Parameter: {param_name}, Shape: {param_tensor.shape}")
    #     print(param_name)
        

if __name__ == "__main__":
    # Replace 'path/to/your/model.bin' with the actual path to your .bin file
    # model_path = 'path/to/your/model.bin'
    print_model_params()
