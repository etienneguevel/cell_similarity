from omegaconf import OmegaConf

import torch
from cell_similarity.model import make_encoder
from cell_similarity.data.loaders import make_datasets, make_dataloaders

cfg_path = '../resources/config_test_dinov2.yaml'


def test_build_encoder():
    cfg = OmegaConf.load(cfg_path)
    encoder = make_encoder(cfg.encoder)


def test_run_encoder():

    dataset_test_path = '../resources/dataset_test'
    datasets = make_datasets(root=dataset_test_path)
    dataloaders = make_dataloaders(datasets, 32)

    cfg = OmegaConf.load(cfg_path)
    encoder = make_encoder(cfg.encoder)

    for batch, labels in dataloaders['train']:
        break

    emb = encoder(batch)
    assert emb.shape[-1] == encoder.embed_dim

if __name__ == '__main__':
    test_build_encoder()
    if torch.cuda.is_available():
        test_run_encoder()

