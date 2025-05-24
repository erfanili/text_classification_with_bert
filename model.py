import torch.nn as nn
from transformers import DistilBertModel
from lightning import LightningModule
from torchmetrics.classification import Accuracy
import torch

class Classifier(LightningModule):
    def __init__(self,model,lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768,100),
            nn.ReLU(),
            nn.Linear(100,5)
        )
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass",num_classes=5)
        n_total, n_trainable = 0, 0
        for p in self.parameters():
            n_total += p.numel()
            if p.requires_grad:
                n_trainable += p.numel()
        print(f"Total parameters: {n_total:,}")
        print(f"Trainable parameters: {n_trainable:,}")
                
    def forward(self,input_ids,attention_mask):
        x = self.model(input_ids,attention_mask).last_hidden_state[:,0]
        return self.classifier(x)
    
    
    def step(self, batch, stage):
        input_ids,attention_mask, labels = batch
        logits = self(input_ids,attention_mask)
        loss = self.loss_fn(logits,labels)
        acc = self.acc(logits, labels)
        self.log(f'{stage}_loss',loss, prog_bar=True)
        self.log(f'{stage}_acc', acc, prog_bar=True)
        
        return loss
    
    def training_step(self, batch,batch_idx):
        return self.step(batch,'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')
    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr =self.hparams.lr)
        