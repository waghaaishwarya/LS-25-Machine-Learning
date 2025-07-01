# -*- coding: utf-8 -*-

# importing necessary libraries
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np

# load imdb data using hugging face datasets library
dataset = load_dataset("imdb")

# tokenizer and preprocessing
# load the tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# tokenize the dataset (truncating long sequence to fit BERT's input limit)
def tokenize_fn(example):
  return tokenizer(example['text'], truncation=True)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# load pre-trained bert model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)

# metrics: accuracy and f1 score
def compute_metrics(pred):
  labels = pred.label_ids
  preds = np.argmax(pred.predictions, axis=1)
  return {
      "accuracy": accuracy_score(labels, preds),
      "f1": f1_score(labels, preds, average = "macro")
  }

# training: setting up training arguments
args = TrainingArguments(
    output_dir = "./results",
    eval_strategy = "epoch",  # evaluate at end of each epoch
    save_strategy = "epoch",   # save model after each epoch
    num_train_epochs = 4,  # no. of training epochs
    per_device_train_batch_size= 8,  # batch size for training
    per_device_eval_batch_size = 8,  # batch size for evaluation
    load_best_model_at_end = True, # keep best model, based on val loss
    logging_dir = "./logs"
)

# initialize hugging face trainer
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_dataset["train"].shuffle(seed=42),
    eval_dataset = tokenized_dataset["test"].select(range(1000)), # subset for faster validation
    tokenizer = tokenizer,
    data_collator = DataCollatorWithPadding(tokenizer),
    compute_metrics = compute_metrics,
)

# train model
trainer.train()

#inference function for predicting sentiment of new text
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    return "positive" if predicted_class == 1 else "negative"

#run a sample inference
sample_text = "The movie was absolutely fantastic!"
print(f"Sample text: {sample_text}")
print("Predicted sentiment:", predict_sentiment(sample_text))