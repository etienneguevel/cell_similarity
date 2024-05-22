from omegaconf import OmegaConf

import torch
from cell_similarity.train.train import main as train_main

cfg_path = '../resources/config_test_dinov2.yaml'

def test_train():

    cfg = OmegaConf.load(cfg_path)
    train_main(cfg)

if __name__ == '__main__':
    if torch.cuda.is_available():
        test_train()