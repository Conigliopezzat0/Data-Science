from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Scarica le risorse necessarie di NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Funzione di pulizia del testo
def clean_text(text):
    if isinstance(text, float):
        return ""
    text = re.sub(r'<.*?>', '', text)   # Rimuove HTML tags
    text = re.sub(r'\W', ' ', text)     # Rimuove caratteri speciali
    text = re.sub(r'\s+', ' ', text)    # Rimuove spazi multipli
    text = re.sub(r'\d', '', text)      # Rimuove numeri
    text = text.strip()                 # Rimuove spazi all'inizio e alla fine
    return text

# Tokenizzazione e rimozione delle stopwords
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# Stemming
stemmer = PorterStemmer()

def stem_tokens(tokens):
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens

# Preprocessamento del testo
def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = tokenize_and_remove_stopwords(cleaned_text)
    stemmed_tokens = stem_tokens(tokens)
    return ' '.join(stemmed_tokens)

# Crea l'app Flask
app = Flask(__name__)

# Carica il modello SVM e il vettorizzatore TF-IDF
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Definisci l'endpoint di predizione
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    email_text = data['email_text']
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return jsonify({'phishing': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
