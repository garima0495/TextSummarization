# Setup and Install Libraries
!pip install transformers datasets rouge-score nltk torch bert-score

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Load the CNN/DailyMail Dataset
data_files = {
    "train": "train.csv",
    "test": "test.csv",
    "validation": "validation.csv"
}
dataset = load_dataset('csv', data_files=data_files)

# Preprocess the Data for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    inputs = tokenizer(examples['article'], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples['highlights'], max_length=128, truncation=True, padding="max_length")
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['article', 'highlights', 'id'])

# Split the tokenized datasets
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # Using a smaller subset for demonstration
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

# Configure and Train the BERT Model
model = BertModel.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluation
# Create a pipeline for generating text (summaries)
summary_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)  # Set device to 0 for GPU

# Function to generate summaries
def generate_summary(text):
    return summary_pipeline(text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['generated_text']

# Applying the function to generate summaries on a subset of the test dataset
test_samples = test_dataset.shuffle(seed=42).select(range(50))  # Using a smaller subset for quick evaluation
generated_summaries = [generate_summary(article) for article in test_samples['article']]

# Reference summaries
reference_summaries = test_samples['highlights']

# Compute BERTScore
P, R, F1 = bert_score(generated_summaries, reference_summaries, lang='en')

# Compute ROUGE scores
rouge = load_metric('rouge')
rouge_scores = rouge.compute(predictions=generated_summaries, references=reference_summaries)

# Compute BLEU score
bleu_score = corpus_bleu([[r.split()] for r in reference_summaries], [g.split() for g in generated_summaries], smoothing_function=SmoothingFunction().method1)

# Display the metrics
print("BERTScore Precision:", P.mean().item())
print("BERTScore Recall:", R.mean().item())
print("BERTScore F1 Score:", F1.mean().item())
print("ROUGE-1:", rouge_scores['rouge1'].mid.fmeasure)
print("ROUGE-2:", rouge_scores['rouge2'].mid.fmeasure)
print("ROUGE-L:", rouge_scores['rougeL'].mid.fmeasure)
print("BLEU Score:", bleu_score)
