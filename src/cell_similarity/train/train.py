import argparse

import lightning as L
import torch
from cell_similarity.data import make_dataloaders, make_datasets
from cell_similarity.model import LinearClassifier, make_encoder
from cell_similarity.utils.config import setup

   

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Classifier training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="", metavar="FILE", help="path to the output dir")

    return parser

def main(cfg):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # make the datasets & dataloaders
    datasets = make_datasets(cfg.train.dataset_path)
    dataloaders = make_dataloaders(datasets, batch_size=cfg.train.batch_size)

    # make the model
    encoder = make_encoder(cfg.encoder)
    model = LinearClassifier(
        encoder=encoder,
        num_classes=datasets["train"].num_classes
    )
    model = model.to(device)
    model.train()

    # freeze the layers of the encoder (we only train the projection layer)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # instantiate the Trainer
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=cfg.train.output_dir)
    trainer = L.Trainer(max_epochs=cfg.train.num_epochs, callbacks=[checkpoint_callback], accelerator=device)

    # train
    trainer.fit(model=model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["validation"])

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    cfg = setup(args)
    main(cfg)
