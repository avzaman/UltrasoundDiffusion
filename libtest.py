import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.current_device())  # The index of the currently active GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Name of the GPU
