import streamlit as st
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


nltk.download('punkt')


model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def preprocess_and_tokenize(question):
    tokens = nltk.word_tokenize(question.lower())
    return ' '.join(tokens)

def find_unique_questions(questions):

    processed_questions = [preprocess_and_tokenize(q) for q in questions]

    embeddings = model.encode(processed_questions, convert_to_tensor=True)

    cosine_matrix = cosine_similarity(embeddings)

    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.4, linkage='average', metric='precomputed')
    clusters = clustering_model.fit_predict(1 - cosine_matrix)

    unique_questions = defaultdict(list)
    for idx, label in enumerate(clusters):
        unique_questions[label].append(questions[idx])

    return [cluster[0] for cluster in unique_questions.values()]

st.title("Unique Question Finder")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    # Read the file content
    questions = uploaded_file.read().decode('utf-8').splitlines()
    questions = [q.strip() for q in questions if q.strip()]

    unique_questions = find_unique_questions(questions)

    st.header("Unique Questions")
    for i, question in enumerate(unique_questions):
        st.write(f"Question {i+1}: {question}")
        # st.write({question})

