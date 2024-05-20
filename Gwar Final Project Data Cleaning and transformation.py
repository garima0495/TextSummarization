#!/usr/bin/env python
# coding: utf-8

# # Gwar Project

# In[1]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tarfile
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import datasets
from datasets import load_dataset
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from multiprocessing import Pool
from nltk.stem import WordNetLemmatizer


# # Loading CNN Dataset

# In[2]:


directory_path = '/Users/garimasingh/Desktop/Data Analyst Process/Project/Parquet/'
file_name = 'CNNdataset.parquet'
file_path = os.path.join(directory_path, file_name)

df = pd.read_parquet(file_path)


# In[3]:


df.head(15)


# In[4]:


df.info()


# # Data Cleaning

# In[5]:


# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))  
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Remove rows where any column is null (if null values are not expected)
df = df.dropna()

# 3. Text normalization - removing special characters and trimming excess whitespace
df['article'] = df['article'].str.replace('[^\w\s]', '', regex=True).str.strip()
df['highlights'] = df['highlights'].str.replace('[^\w\s]', '', regex=True).str.strip()

# 4. Remove stopwords from 'article' and 'highlights'
df['article'] = df['article'].apply(remove_stopwords)
df['highlights'] = df['highlights'].apply(remove_stopwords)


# In[6]:


df.head(15)


# # Lemmetizing

# In[7]:


lemmatizer = WordNetLemmatizer()

def lemmetize(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


# In[8]:


df['article'] = df['article'].apply(lemmetize)
df['highlights'] = df['highlights'].apply(lemmetize)


# In[9]:


df.head(15)


# In[10]:


# Remove "CNN" from headlines
df['article'] = df['article'].str.replace('^CNN', '', regex=True)
df['article'] = df['article'].str.strip()


# In[11]:


df.head(15)


# # Loading Xsum Dataset

# In[12]:


_CITATION = """
@article{Narayan2018DontGM,
  title={Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization},
  author={Shashi Narayan and Shay B. Cohen and Mirella Lapata},
  journal={ArXiv},
  year={2018},
  volume={abs/1808.08745}
}
"""

_DESCRIPTION = """
Extreme Summarization (XSum) Dataset.
There are three features:
  - document: Input news article.
  - summary: One sentence summary of the article.
  - id: BBC ID of the article.
"""

# From https://github.com/EdinburghNLP/XSum/issues/12
_URL_DATA = "/Users/garimasingh/Desktop/Data Analyst Process/Project/Parquet/XSUM-EMNLP18-Summary-Data-Original.tar.gz"
_URL_SPLITS = (
    "https://raw.githubusercontent.com/EdinburghNLP/XSum/master/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
)

_DOCUMENT = "document"
_SUMMARY = "summary"
_ID = "id"

_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
    ]
)


class Xsum(datasets.GeneratorBasedBuilder):
    """Extreme Summarization (XSum) Dataset."""

    # Version 1.2.0 expands coverage, includes ids, and removes web contents.
    VERSION = datasets.Version("1.2.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _DOCUMENT: datasets.Value("string"),
                    _SUMMARY: datasets.Value("string"),
                    _ID: datasets.Value("string"),
                }
            ),
            supervised_keys=(_DOCUMENT, _SUMMARY),
            homepage="https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        files_to_download = {"data": _URL_DATA, "splits": _URL_SPLITS}
        downloaded_files = dl_manager.download(files_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split_path": downloaded_files["splits"],
                    "split_name": "train",
                    "data_dir": "bbc-summary-data",
                    "files": dl_manager.iter_archive(downloaded_files["data"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split_path": downloaded_files["splits"],
                    "split_name": "validation",
                    "data_dir": "bbc-summary-data",
                    "files": dl_manager.iter_archive(downloaded_files["data"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split_path": downloaded_files["splits"],
                    "split_name": "test",
                    "data_dir": "bbc-summary-data",
                    "files": dl_manager.iter_archive(downloaded_files["data"]),
                },
            ),
        ]

    def _generate_examples(self, split_path, split_name, data_dir, files):
        """Yields examples."""

        with open(split_path, "r", encoding="utf-8") as f:
            split_ids = json.load(f)
        split_ids = {k: set(v) for k, v in split_ids.items()}

        for path, f in files:
            if not split_ids[split_name]:
                break
            elif path.startswith(data_dir) and path.endswith(".summary"):
                i = os.path.basename(path).split(".")[0]
                if i in split_ids[split_name]:
                    split_ids[split_name].remove(i)
                    text = "".join(
                        [
                            line.decode("utf-8")
                            for line in f.readlines()
                            if line.decode("utf-8") not in _REMOVE_LINES and line.strip()
                        ]
                    )
                    # Each file follows below format:
                    # [SN]URL[SN]
                    # http://somelink
                    #
                    # [SN]TITLE[SN]
                    # some intro
                    #
                    # [SN]FIRST-SENTENCE[SN]
                    # some intro
                    #
                    # [SN]RESTBODY[SN]
                    # text line.
                    # another text line.
                    # "another text line."

                    # According to the following issue, FIRST-SENTENCE
                    # is the reference summary and TITLE is unused:
                    # https://github.com/EdinburghNLP/XSum/issues/22
                    segs = text.split("[SN]")
                    yield i, {_DOCUMENT: segs[8].strip(), _SUMMARY: segs[6].strip(), _ID: i}


# In[13]:


# Load the Xsum dataset
xsum_dataset = load_dataset("xsum")

# Convert the dataset splits to pandas DataFrames
train_df = pd.DataFrame(xsum_dataset["train"])
validation_df = pd.DataFrame(xsum_dataset["validation"])
test_df = pd.DataFrame(xsum_dataset["test"])

# Display information about the DataFrames
print("Train Dataset Info:")
print(train_df.info())
print("\nValidation Dataset Info:")
print(validation_df.info())
print("\nTest Dataset Info:")
print(test_df.info())


# In[14]:


xsum_dataset = load_dataset("xsum")

full_df = pd.concat([pd.DataFrame(split) for split in xsum_dataset.values()])

print("Full Dataset Info:")
print(full_df.info())


# In[15]:


full_df.head(15)


# # Cleaning the dataset

# In[16]:


# 1. Remove duplicates
full_df = full_df.drop_duplicates()

# 2. Handle missing values (assuming any row with a missing column should be removed)
full_df = full_df.dropna()

# 3. Text normalization - removing extra spaces and special characters if necessary
full_df['document'] = full_df['document'].str.replace('[^\w\s]', '', regex=True).str.strip()
full_df['summary'] = full_df['summary'].str.replace('[^\w\s]', '', regex=True).str.strip()

# 4. Remove stopwords from 'document' and 'summary'
full_df['document'] = full_df['document'].apply(remove_stopwords)
full_df['summary'] = full_df['summary'].apply(remove_stopwords)


# In[17]:


full_df.head(15)


# # Lemmetize

# In[18]:


full_df['document'] = full_df['document'].apply(lemmetize)
full_df['summary'] = full_df['summary'].apply(lemmetize)


# In[19]:


full_df.head(15)


# # Regularization

# In[20]:


def regularize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Optional: remove digits
    # text = re.sub(r'\d+', '', text)
    return text

# Apply regularization to both dataframes
df['article'] = df['article'].apply(regularize_text)
df['highlights'] = df['highlights'].apply(regularize_text)
full_df['document'] = full_df['document'].apply(regularize_text)
full_df['summary'] = full_df['summary'].apply(regularize_text)


# In[21]:


df.head() #CNNDataset


# In[22]:


full_df.head() #XsumDataset


# # Transformation of Data

# In[23]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error


# In[25]:


# Apply TF-IDF Vectorization across both DataFrames with the same vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
df_tfidf = tfidf_vectorizer.fit_transform(df['article'])
full_df_tfidf = tfidf_vectorizer.transform(full_df['document']) 
print("Numerical conversion of df:",df_tfidf.shape)
print("Numerical conversion of full_df:",full_df_tfidf.shape)

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=100)
df_reduced = svd.fit_transform(df_tfidf)  
full_df_reduced = svd.transform(full_df_tfidf)  

print("Reduced shape for df:", df_reduced.shape)
print("Reduced shape for full_df:", full_df_reduced.shape)


# In[26]:


# Adding new features such as text length
df['article_length'] = df['article'].apply(len)
df['highlights_length'] = df['highlights'].apply(len)
full_df['document_length'] = full_df['document'].apply(len)
full_df['summary_length'] = full_df['summary'].apply(len)


# In[27]:


df.head()


# In[28]:


full_df.head()


# # Splitting datasets into test, train and validation

# CNN Dataset

# In[29]:


# Splitting data into training set (90%) and the remaining 10%
train, remaining = train_test_split(df, test_size=0.1, random_state=42)

# Splitting the remaining data into validation and test sets (50% each of remaining data, which equals 5% each of total data)
validation, test = train_test_split(remaining, test_size=0.5, random_state=42)

# Saving datasets to CSV files
train.to_csv('/Users/garimasingh/Desktop/Data Analyst Process/Project/CNN Dataset/CNNTrain.csv', index=False)
validation.to_csv('/Users/garimasingh/Desktop/Data Analyst Process/Project/CNN Dataset/CNNValidation.csv', index=False)
test.to_csv('/Users/garimasingh/Desktop/Data Analyst Process/Project/CNN Dataset/CNNTest.csv', index=False)


# Xsum dataset

# In[30]:


# Splitting data into training set (90%) and the remaining 10%
train, remaining = train_test_split(full_df, test_size=0.1, random_state=42)

# Splitting the remaining data into validation and test sets (50% each of remaining data, which equals 5% each of total data)
validation, test = train_test_split(remaining, test_size=0.5, random_state=42)

# Define the path to save the files
output_path = '/Users/garimasingh/Desktop/Data Analyst Process/Project/Xsum dataset/'

# Saving datasets to CSV files
train.to_csv(output_path + 'XsumTrain.csv', index=False)
validation.to_csv(output_path + 'XsumValidation.csv', index=False)
test.to_csv(output_path + 'XsumTest.csv', index=False)

