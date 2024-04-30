import torch
from cell_similarity.utils.memory import *

def main():

    device = device = "gpu" if torch.cuda.is_available() else "cpu"

    if device == "gpu":
        print_gpu_utilization()

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.to(device)

    if device == "gpu":
        print_gpu_utilization()

if __name__ == 'main':
    main()