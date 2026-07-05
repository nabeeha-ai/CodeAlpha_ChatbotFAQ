import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

faq_data = {
    "what is artificial intelligence (AI)": "AI is the simulation of human intelligence in machines that think and learn like humans.",
    "what is machine learning (ML)": "Machine Learning is a subset of AI that enables systems to learn from experience without being explicitly programmed.",
    "what is deep learning (DL)": "Deep Learning uses neural networks with many layers to analyze various factors of data.",
    "what is python": "Python is a high-level programming language widely used in AI and data science.",
    "what is neural network ": "A neural network mimics the human brain to recognize patterns and relationships in data.",
    "what is data science": "Data Science extracts knowledge and insights from structured and unstructured data.",
    "what is nlp natural language processing": "NLP helps computers understand, interpret and manipulate human language.",
    "what is computer vision": "Computer Vision trains computers to interpret and understand images and videos.",
    "what is chatbot": "A chatbot is an AI program designed to simulate conversation with human users.",
    "what is reinforcement learning (RL)": "Reinforcement Learning trains an agent by rewarding correct decisions and penalizing wrong ones."
}

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

questions = list(faq_data.keys())
answers = list(faq_data.values())
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("FAQ Chatbot")
st.write("Ask me anything about AI & Data Science!")

user_input = st.text_input("Your question:")

if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please enter a question!")
    else:
        processed_input = preprocess(user_input)
        user_vector = vectorizer.transform([processed_input])
        similarities = cosine_similarity(user_vector, question_vectors)
        best_idx = similarities.argmax()
        best_score = similarities[0][best_idx]
        if best_score < 0.1:
            st.error("Sorry, I don't have an answer for that. Try rephrasing!")
        else:
            st.success(answers[best_idx])