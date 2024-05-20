# Setup and Install Libraries
!pip install transformers datasets rouge-score nltk torch bert-score

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, pipeline
from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Load the XSum Dataset
data_files = {
    "train": "train.csv",
    "test": "test.csv",
    "validation": "validation.csv"
}
dataset = load_dataset('csv', data_files=data_files)

# Preprocess the Data for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    # Encode the documents but do not set a maximum length to allow full encoding
    inputs = tokenizer(examples['document'], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    # Assume the summaries are also processed similarly but check for length constraints
    outputs = tokenizer(examples['summary'], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': outputs['input_ids']}

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Configure and Train the BERT Model
model = BertModel.from_pretrained('bert-base-uncased')

# Define a method to extract features using BERT
def extract_features(dataset):
    # Extract embeddings from BERT
    inputs = {'input_ids': dataset['input_ids'], 'attention_mask': dataset['attention_mask']}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state

# Example of extracting features (use torch.no_grad() to reduce memory usage)
features = extract_features(tokenized_datasets['train'][:10])

# Evaluation
# Create a pipeline for generating text (summaries)
summary_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)  

# Function to generate summaries
def generate_summary(text):
    return summary_pipeline(text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['generated_text']

# Applying the function to generate summaries on a subset of the test dataset
test_samples = encoded_dataset['test'].shuffle(seed=42).select(range(50))  # Using a smaller subset for quick evaluation
generated_summaries = [generate_summary(article['input_ids']) for article in test_samples]

# Reference summaries
reference_summaries = test_samples['summary']

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
