import os
from pathlib import Path
import argparse

import lightning as L
import torch
import tqdm
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from cell_similarity.data import make_dataloaders, make_datasets
from cell_similarity.data.datasets import EmbeddingDataset
from cell_similarity.model import make_encoder, LinearProjection
from cell_similarity.utils.config import setup

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Classifier training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="", metavar="FILE", help="path to the output dir")

    return parser

def main(cfg):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # make the datasets & dataloaders
    datasets = make_datasets(root=cfg.preprocess.dataset_path)
    dataloaders = make_dataloaders(datasets, batch_size=cfg.preprocess.batch_size)

    num_classes = datasets["train"].num_classes

    # load the encoder
    encoder = make_encoder(cfg.encoder)
    encoder = encoder.to(device)

    # process the batches
    for state in dataloaders.keys():
        embeddings = torch.tensor([], dtype=torch.float32)
        labels = torch.tensor([])
        print(f"Starting to preprocess the {state} data.")
        for batch, lab in tqdm(dataloaders[state]):
            batch, lab = batch.to(device), lab.to(device)

            with encoder.no_grad():
                embs = encoder(batch)
            
            embeddings = torch.stack([embeddings, embs])
            labels = torch.stack([labels, lab])

        save_path = Path(cfg.preprocess.cache_path) / state

        print(f"Saving at {save_path}")

        os.makedirs(save_path, exist_ok=True)
        torch.save(embeddings, save_path / 'embeddings.pt')
        torch.save(labels, save_path / 'labels.pt')
        del(embeddings)
        del(labels)

    # initialize the datasets for the linear classifier training
    datasets = {EmbeddingDataset(Path(cfg.preprocess.cache_path) / state) for state in dataloaders.keys()}
    dataloaders = make_dataloaders(datasets, batch_size=cfg.train.batch_size)

    # initialize the model
    kwargs = cfg.model

    model = LinearProjection(num_classes=num_classes, **kwargs)
    model.train()

    # instantiate the Trainer
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.train.output_dir, monitor="val_loss", save_top_k=1, save_last=True)
    early_stopping = EarlyStopping("val_loss", patience=1000)
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs, callbacks=[checkpoint_callback, early_stopping], accelerator=device)

    trainer.fit(model=model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["validation"])


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    cfg = setup(args)
    main(cfg)