from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

app = Flask(__name__)


# Function to read preprocessed corpus texts from a text file
def read_preprocessed_corpus(file_path):
    with open(file_path, 'r') as file:
        corpus = file.readlines()
    return [text.strip() for text in corpus]


# Load preprocessed corpus from a text file
corpus_file_path = 'preprocessed_corpus.txt'
corpus = read_preprocessed_corpus(corpus_file_path)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess text: Lowercasing, remove punctuation, stopwords, and stemming
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word not in string.punctuation]
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Function to compute similarity score between two texts using BERT embeddings
def similarity_score(text1, text2):
    def get_bert_embedding(text):
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(input_ids)
            last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :].squeeze().numpy()
        return cls_embedding

    embedding1 = get_bert_embedding(text1)
    embedding2 = get_bert_embedding(text2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity


# Function to compute similarity scores between input text and corpus
def compute_similarity(input_text, corpus):
    input_text = preprocess_text(input_text)
    similarity_scores = []
    for text in corpus:
        similarity = similarity_score(input_text, text)
        similarity_scores.append(similarity)
    return similarity_scores


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    text = request.form['text']
    similarity_scores = compute_similarity(text, corpus)

    # Check if any similarity score crosses a threshold for plagiarism
    threshold = 0.8
    plagiarism_detected = any(score > threshold for score in similarity_scores)

    # Calculate the maximum similarity score and convert it to a percentage
    max_similarity_score = max(similarity_scores) if similarity_scores else 0
    plagiarism_percentage = round(max_similarity_score * 100, 2)

    if plagiarism_detected:
        plagiarism_result = f"Plagiarism detected! Similarity percentage: {plagiarism_percentage}%"
    else:
        plagiarism_result = "No plagiarism detected. The text is original."

    return render_template('result.html', result=plagiarism_result)


if __name__ == '__main__':
    app.run(debug=True)
