import torch
import torch.nn as nn
import lightning as L

class LinearClassifier(L.LightningModule):
    def __init__(self, encoder: nn.Module, projection: nn.Module = None, num_classes: int = None):
        super().__init__()
        self.encoder = encoder
        self.embedding_size = encoder.embed_dim
        self.projection = projection
        self.num_classes = num_classes
        if self.projection is None:
            if self.num_classes is None:
                raise RuntimeError("no projection is given, nor the number of classes to build the basic projector -> can't build classifier.")
            self.projection = self.build_base_projection()

    def build_base_projection(self):

        return nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.num_classes),
            nn.Softmax()
        )
    
    def forward(self, x):
        cls = self.encoder(x)
        z = self.projection(cls)

        return z
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)

        loss = nn.functional.cross_entropy(z, y)
        self.log("train loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        _, preds = torch.max(z, 1)
        acc = torch.sum(y == preds) / len(y)
        self.log("train_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)

        val_loss = nn.functional.cross_entropy(z, y)
        self.log("val_loss", val_loss, prog_bar=True)

        _, preds = torch.max(z, 1)
        acc = torch.sum(y == preds) / len(y)
        self.log("val_accuracy", acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.projection.parameters(), lr=1e-3)
        
        return optimizer
    
