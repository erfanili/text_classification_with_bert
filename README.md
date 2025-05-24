# Text Classification with BERT

This project fine-tunes a pre-trained DistilBERT model for binary sentiment classification on Amazon product reviews using PyTorch Lightning and Hugging Face Transformers.

## Dataset

- **Source**: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Subset**: `raw_review_All_Beauty`
- **Labeling rule**:
  - Ratings 4 and 5 → `label = 1` (positive)
  - Ratings 1, 2, 3 → `label = 0` (not positive)

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py \
  --model_name distilbert-base-uncased \
  --dataset McAuley-Lab/Amazon-Reviews-2023 \
  --subset raw_review_All_Beauty \
  --batch_size 32 \
  --lr 2e-5 \
  --epochs 3 \
  --seed 42 \
  --use_wandb
```