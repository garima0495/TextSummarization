# -*- coding: utf-8 -*-
"""CNN_OpenAI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13BOrCERuAjTih-DWJ1AS5lWfpfyUgvAu
"""

!pip install openai

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW
from datasets import load_dataset
import numpy as np

data_files = {
    "train": "CNNTrain.csv",
    "test": "CNNTest.csv",
    "validation": "CNNValidation.csv"
}

dataset = load_dataset('csv', data_files=data_files)

# Preprocessing the data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def preprocess_data(examples):
    return tokenizer(examples['article'], truncation=True, padding='max_length', max_length=512)

train_encodings = train_dataset.map(preprocess_data, batched=True)

# Prepare the dataset for PyTorch
def format_dataset(examples):
    input_ids = torch.tensor(examples['input_ids'])
    attention_mask = torch.tensor(examples['attention_mask'])
    labels = torch.tensor([1 if examples['highlights'] else 0])  # Dummy labels for example
    return TensorDataset(input_ids, attention_mask, labels)

train_dataset = format_dataset(train_encodings)

# DataLoader
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)

# Load model with LoRA and PEFT adaptations
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
model.resize_token_embeddings(len(tokenizer))

# Adding LoRA layers (simplified example)
class LoRALayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.r = torch.nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))

    def forward(self, hidden_states):
        return self.dense(hidden_states) + self.r @ hidden_states

model.roberta.encoder.layer[0].intermediate = LoRALayer(model.config)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(3):  # num_epochs
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

import openai

# Replace 'your_api_key' with your actual API key
api_key = 'sk-proj-7c2PVoOmKXkGcSMMyZ9AT3BlbkFJN17FfVDMnrQQnQ7Bo8fP'
openai.api_key = api_key

!pip install datasets

import pandas as pd
test_df = pd.DataFrame(dataset['test'])

test_df.head()

test_df.drop(columns=['id'], inplace=True)

test_df.head()

test_df['article'][0]

test_df['highlights'][0]

!pip install openai==0.28

pip show openai

import time

# Define a function to generate summaries using the text-davinci-003 model
import openai

def generate_summary_with_openai(article_text):
    prompt = f"Summarize this article:\n{article_text}"

    # Updated method to use the Chat Completion endpoint
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Updated model name for better performance
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=150  # Adjusted token limit based on requirements
    )

    # Adjusted extraction of summary based on the new output format
    summary = response['choices'][0]['message']['content']
    return summary.strip()


# Create an empty column 'model_generated_openai' in test_df to store the generated summaries
test_df['model_generated'] = ""

# Define the batch size and delay
batch_size = 3
delay_seconds = 60  # 1 minute

# Generate summaries in batches
for batch_start in range(0, 25, batch_size):
    batch_end = min(batch_start + batch_size, 25)  # Ensure the end does not exceed 25
    articles_to_process = test_df['article'][batch_start:batch_end]

    # Generate summaries for the batch
    generated_summaries = []
    for article_text in articles_to_process:
        summary = generate_summary_with_openai(article_text)
        generated_summaries.append(summary)

    # Store the generated summaries in the DataFrame
    test_df.loc[batch_start:batch_end-1, 'model_generated'] = generated_summaries

    # Introduce a delay after each batch
    if batch_end < 25:
        print(f"Generated summaries for articles {batch_start+1}-{batch_end}. Waiting for {delay_seconds} seconds before the next batch...")
        time.sleep(delay_seconds)

# Display the updated DataFrame with generated summaries using OpenAI's model
print(test_df[['article', 'model_generated']])

test_df.head(25)

test_df['model_generated'][4]

test_df['model_generated'][5]

!pip install rouge

from rouge import Rouge

# Initialize the ROUGE evaluator
rouge = Rouge()

# Select the first 25 rows of your DataFrame for evaluation
num_samples = 25
sampled_df = test_df.head(num_samples)

# Extract the generated summaries and reference summaries for the selected samples
generated_summaries = sampled_df['model_generated'].tolist()
reference_summaries = sampled_df['highlights'].tolist()

# Calculate ROUGE scores for the selected samples
rouge_scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)

# Print the ROUGE scores
print("ROUGE Scores:", rouge_scores)

from nltk.translate.bleu_score import corpus_bleu

# Select the first 25 rows of your DataFrame for evaluation
num_samples = 25
sampled_df = test_df.head(num_samples)

# Extract the generated summaries and reference summaries for the selected samples
generated_summaries = sampled_df['model_generated'].tolist()
reference_summaries = sampled_df['highlights'].tolist()

# Calculate BLEU score for the selected samples
bleu_score = corpus_bleu(reference_summaries, generated_summaries)
print("BLEU Score for 15 Summaries:", bleu_score)

!pip install bert_score

from bert_score import score
# Select the first 25 rows of your DataFrame for evaluation
num_samples = 25
sampled_df = test_df.head(num_samples)

# Extract the generated summaries and reference summaries for the selected samples
generated_summaries = sampled_df['model_generated'].tolist()
reference_summaries = sampled_df['highlights'].tolist()

# Calculate BERT Score
P, R, F1 = score(generated_summaries, reference_summaries, lang="en", verbose=True)

# Print BERT Score
print("BERT Precision:", P.mean().item())
print("BERT Recall:", R.mean().item())
print("BERT F1 Score:", F1.mean().item())

test_df['model_generated'][5]