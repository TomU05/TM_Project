import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# Load your CSV
df = pd.read_csv("youtube_comments.csv", quotechar='"')

# Convert boolean labels to integers (0 for False, 1 for True)
df['sarcastic'] = df['sarcastic'].astype(int)

# Basic cleaning
df = df[['comment', 'sarcastic']].dropna()

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['sarcastic'], random_state=123)

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# Load tokenizer - switched to RoBERTa
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization
def tokenize(batch):
    return tokenizer(batch['comment'], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Rename columns
train_ds = train_ds.rename_column("sarcastic", "labels")
test_ds = test_ds.rename_column("sarcastic", "labels")
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Calculate improved class weights
neg_count = (train_df['sarcastic'] == 0).sum()
pos_count = (train_df['sarcastic'] == 1).sum()
total = neg_count + pos_count
smoothing = 0.1  # Prevents extreme weights
pos_weight = (total - pos_count + smoothing) / (pos_count + smoothing)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Move to device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Custom Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean()

# Initialize loss with class weights
loss_fn = FocalLoss(alpha=pos_weight)

# Custom Trainer
class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Metrics with focus on sarcastic class
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_1": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_1": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_1": f1_score(labels, preds, pos_label=1, zero_division=0),
    }

# Improved training arguments
training_args = TrainingArguments(
    output_dir="./sarcasm_model",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_1",
    greater_is_better=True,
)

# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    loss_fn=loss_fn,
)

# Train
trainer.train()

# Final evaluation
eval_results = trainer.evaluate()
print(eval_results)

# Detailed analysis
from sklearn.metrics import classification_report
preds = trainer.predict(test_ds)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Not Sarcastic", "Sarcastic"]))