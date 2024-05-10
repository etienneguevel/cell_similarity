import torch
from cell_similarity.utils.memory import *

def main():

    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    print("The device used for this test is:", device)
    if device == "cuda":
        print("Before loading the model")
        print_gpu_utilization()

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.to(device)

    if device == "cuda":
        print("After loading the model")
        print_gpu_utilization()

if __name__ == '__main__':
    main()
