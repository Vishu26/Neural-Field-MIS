import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torchmetrics import MeanSquaredError


def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class ImageFitting(torch.utils.data.Dataset):
    def __init__(self, img, p):
        super().__init__()
        self.pixels = img.view(-1, 1)
        self.coords = get_mgrid(p, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels


class ImagePrediction(torch.utils.data.Dataset):
    def __init__(self, p):
        super().__init__()
        self.coords = get_mgrid(p, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
            
        return self.coords

class NeuralField(pl.LightningModule):
  def __init__(self, dataloader, n, p, activation='sigmoid'):
    super().__init__()
    self.dataloader = dataloader
    self.p = p

    if activation == 'sigmoid':
        self.activation = nn.Sigmoid()
    elif activation == 'relu':
        self.activation = nn.ReLU()
    elif activation == 'tanh':
        self.activation = nn.Tanh()
    elif activation == 'sine':
        self.activation = torch.sin
    else:
        raise ValueError("Activation function not supported") 
    self.lin1 = nn.Linear(2, n)
    self.lin2 = nn.Linear(n, n)
    self.lin3 = nn.Linear(n, n)
    self.lin4 = nn.Linear(n, n)
    self.lin5 = nn.Linear(n, 1)
    self.out = nn.Tanh()
    self.criterion = MeanSquaredError()
  
  def forward(self, x):
    x = self.activation(self.lin1(x))
    x = self.activation(self.lin2(x))
    x = self.activation(self.lin3(x))
    x = self.activation(self.lin4(x))
    x = self.activation(x)
    return self.out(x)
  
  def shared_step(self, batch):
    x, y = batch
    pred = self(x)
    loss = self.criterion(pred, y)
    if self.current_epoch%50==0:
        plt.imshow(pred.detach().numpy().reshape(self.p, self.p), cmap='gray')
        plt.savefig(F"images/sigmoid/image_{self.current_epoch}.png", dpi=160)
    return loss
  
  def training_step(self, batch, batch_idx):
    loss = self.shared_step(batch)
    return {'loss':loss}
  
  def training_step_end(self, outputs):
    loss = outputs['loss'].mean()
    self.log('train_loss', loss, prog_bar=True, on_epoch=True)
    return {"loss": loss}

  def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        return self(batch[0])
  
  def train_dataloader(self):
    return self.dataloader
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    return optimizer


if __name__ == "__main__":

    img = Image.open("training/image.png").convert('L')
    plt.imshow(img, cmap='gray')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)

    p = 256
    n = 64
    image = ImageFitting(img)
    dataloader = torch.utils.data.DataLoader(image, batch_size=1, pin_memory=True, num_workers=0)

    model = NeuralField(dataloader, n, p, activation='sigmoid')

    trainer = pl.Trainer(max_epochs=2000)

    trainer.fit(model)

    predloader = torch.utils.data.DataLoader(ImagePrediction(256), batch_size=1, pin_memory=True, num_workers=0)
    preds = trainer.predict(model, dataloaders=predloader)
    plt.imshow(preds[0].detach().numpy().reshape(256, 256), cmap='gray')

    predloader = torch.utils.data.DataLoader(ImagePrediction(512), batch_size=1, pin_memory=True, num_workers=0)
    preds = trainer.predict(model, dataloaders=predloader)
    plt.imshow(preds[0].detach().numpy().reshape(512, 512), cmap='gray')

    predloader = torch.utils.data.DataLoader(ImagePrediction(128), batch_size=1, pin_memory=True, num_workers=0)
    preds = trainer.predict(model, dataloaders=predloader)
    plt.imshow(preds[0].detach().numpy().reshape(128, 128), cmap='gray')


