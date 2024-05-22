from omegaconf import OmegaConf

import torch
from cell_similarity.model import LinearClassifier, make_encoder
from cell_similarity.data.loaders import make_datasets, make_dataloaders

cfg_path = '../resources/config_test_dinov2.yaml'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_build_LinearClassifier():
    cfg = OmegaConf.load(cfg_path)
    datasets = make_datasets(cfg.train.dataset_path)
    dataloaders = make_dataloaders(datasets, batch_size=cfg.train.batch_size)

    encoder = make_encoder(cfg.encoder)
    model = LinearClassifier(
        encoder=encoder,
        num_classes=datasets["train"].num_classes
    )
    model = model.to(device)

    for batch, labs in dataloaders["train"]:
        batch, labs = batch.to(device), labs.to(device)
        break
    
    outputs = model(batch)
    print(outputs.shape, labs.shape)
    loss = torch.nn.functional.cross_entropy(outputs, labs)
    print(loss)

if __name__ == "__main__":
    test_build_LinearClassifier()
