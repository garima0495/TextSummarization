# -*- coding: utf-8 -*-
"""Falcon-7b-instructCNNDailyNews

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tMmnS_yyWRDljg2gYzpzzHZHuVPjj-GH
"""

!pip install transformers
!pip install langchain
!pip install datasets
!pip install torch torchvision torchaudio
from datasets import load_dataset

data_files = {
    "train": "CNNTrain.csv",
    "test": "CNNTest.csv",
    "validation": "CNNValidation.csv"
}

dataset = load_dataset('csv', data_files=data_files)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

def preprocess_function(examples):
    inputs = examples['article']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['highlights'], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(
  'tiiuae/falcon-7b-instruct',
  trust_remote_code=True
)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduced batch size
    per_device_eval_batch_size=1,   # Reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=10_000,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# Train the model
trainer.train()

from transformers import pipeline
from langchain import HuggingFacePipeline, PromptTemplate,  LLMChain


# Set up the pipeline for text generation
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100, do_sample=True, use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

# Wrap the pipeline in HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.1})

# Define the prompt template
template = """
Write a concise summary of the following text delimited by triple backquotes.
```{text}```
SUMMARY:
"""

prompt = PromptTemplate(template=template, input_variables=["text"])

# Create LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

df = pd.DataFrame(dataset['test'])[['article', 'highlights','id']]

df.head()

df.drop(columns=['id'], inplace=True)
df.head()

df['model_generated'] = ""

# Define a function to generate summaries and populate the 'model_generated' column
def generate_and_store_summary(row):
    article_text = row['article']
    summary = llm_chain.run(article_text)
    return summary

df['model_generated'] = df.apply(generate_and_store_summary, axis=1)

# Apply the logic to all rows of the 'model_generated' column
df['model_generated'] = df['model_generated'].apply(lambda text: text.split('SUMMARY:')[1].strip() if 'SUMMARY:' in text else text)

print(df[['article', 'model_generated']].head(10))

df['article'][8]

df['model_generated'][8]

!pip install rouge
!pip install nltk
!pip install bert_score

from rouge import Rouge

# Initialize the ROUGE evaluator
rouge = Rouge()

sampled_df = df.dropna(subset=['model_generated', 'highlights'])

# Extract the generated summaries and reference summaries for the selected samples
generated_summaries = sampled_df['model_generated'].tolist()
reference_summaries = sampled_df['highlights'].tolist()

# Calculate ROUGE scores for the selected samples
rouge_scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)

from nltk.translate.bleu_score import corpus_bleu


# Extract the generated summaries and reference summaries for the selected samples
generated_summaries = sampled_df['model_generated'].tolist()
reference_summaries = sampled_df['highlights'].tolist()

# Calculate BLEU score for the selected samples
bleu_score = corpus_bleu(reference_summaries, generated_summaries)

from bert_score import score

# Extract the generated summaries and reference summaries for the selected samples
generated_summaries = sampled_df['model_generated'].tolist()
reference_summaries = sampled_df['highlights'].tolist()

# Calculate BERT Score
P, R, F1 = score(generated_summaries, reference_summaries, lang="en", verbose=True)

# Print BERT Score
print("ROUGE Scores:", rouge_scores)
print("BLEU Score for Summaries:", bleu_score)
print("BERT Precision:", P.mean().item())
print("BERT Recall:", R.mean().item())
print("BERT F1 Score:", F1.mean().item())