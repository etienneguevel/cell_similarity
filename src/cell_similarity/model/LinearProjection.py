import torch
import torch.nn as nn
import lightning as L

class LinearProjection(L.LightningModule):
    def __init__(self, embedding_size, num_classes:int, lr:float=1e-3, gamma:float=0.1, momentum:float=0.9, warmup_epochs:int=100, min_lr:float=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.lr = lr
        self.gamma = gamma
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr  

        self.model = self._build_model()
        self.lr_scheduler = self._build_lr_scheduler()

    def _build_model(self):

        return nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.num_classes),
            # nn.Softmax()
        )
    
    def _build_lr_scheduler(self):
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warm-up
                return (epoch / self.warmup_epochs)
            else:
                # Exponential decay
                lr = self.gamma ** (epoch - self.warmup_epochs)
                # Clamp to minimum learning rate
                return max(lr, self.min_lr / self.lr)
        
        return lr_lambda

    def forward(self, x):
        z = self.model(x)

        return z
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)

        loss = nn.functional.cross_entropy(z, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=self.gamma)

        return [optimizer], [lr_scheduler]
