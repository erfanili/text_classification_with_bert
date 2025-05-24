import argparse 
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from model import Classifier
from data import AmazonReviewDataModule
from transformers import DistilBertTokenizer,DistilBertModel

def main(args):
    seed_everything(args.seed)
    
    data = AmazonReviewDataModule(
        dataset_name = args.dataset,
        subset = args.subset,
        tokenizer= DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
        batch_size = args.batch_size
    )
    
    model =Classifier(
        model = DistilBertModel.from_pretrained(args.model_name),
        lr = args.lr
    )
    
    logger = WandbLogger(project= "middle-ground", log_model = "all")
    
    trainer = Trainer(
        max_epochs = args.epochs,
        logger = logger,
        accelerator = 'auto',
        devices = 1,
        val_check_interval=0.1
    )
    
    trainer.fit(model,data)
    trainer.test(model,data)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="McAuley-Lab/Amazon-Reviews-2023")
    parser.add_argument("--subset", type=str, default="raw_review_All_Beauty")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    
    main(args)