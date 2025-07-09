# -*- coding: utf-8 -*-
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import math
from tqdm import tqdm
import gradio as gr

# pip install -U datasets

# load dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# load tokenizer

tokenizer =  AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.pad_token_id

# tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# group text
block_size = 128

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    input_ids = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
    return {"input_ids": input_ids, "labels": input_ids}

#  remove_columns ensures we drop attention_mask etc.
lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    remove_columns=tokenized_dataset["train"].column_names
)

# data collator for language modelling

data_collator = DataCollatorForLanguageModeling(
    tokenizer= tokenizer,
    mlm=False
)

# training arguments
training_args = TrainingArguments(
    output_dir = "./results",
    overwrite_output_dir = True,
    eval_strategy="epoch",
    per_device_train_batch_size =2,
    per_device_eval_batch_size = 2,
    num_train_epochs = 1,
    weight_decay= 0.01,
    logging_dir = './logs',
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"] ,
    eval_dataset= lm_dataset["validation"],
    data_collator=data_collator,
)

trainer.train()

# evaluation using perplexity

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss'])}")

# top k accuracy

def compute_top_k_accuracy(model, dataset, tokenizer, k=5, max_batches=100):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0
    total = 0

    for i, example in enumerate(tqdm(dataset)):
        if i >= max_batches:
            break

        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
        labels = torch.tensor(example["labels"]).unsqueeze(0)

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        # Ignore padding/label=-100
        for t in range(1, logits.shape[1]):
            if labels[0, t] == -100:
                continue
            top_k = torch.topk(logits[0, t-1], k).indices
            if labels[0, t] in top_k:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0

# validation set (small subset for speed T_T)
top_k_acc = compute_top_k_accuracy(model, lm_dataset["validation"], tokenizer, k=5, max_batches=1930)

print(f"Top-5 Accuracy: {top_k_acc:.4f}")

# gradio interface

def complete_text(prompt, max_new_tokens=3):
    if not prompt.strip():
        return "Please enter some text!"

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        return f"Error: {str(e)}"

gr.Interface(
    fn=complete_text,
    inputs=["text", gr.Slider(1, 10, value=3, label="Number of Words")],
    outputs="text",
    title="Next Word Predictor",
    description="give a sentence prompt & predict next few words using your fine-tuned GPT-2 model"
).launch()

