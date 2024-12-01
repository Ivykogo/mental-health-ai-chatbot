import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import logging
from flask import Flask, render_template, request, Blueprint

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load models and data
def load_data():
    try:
        intents = json.loads(open('intents.json').read())
        words = pickle.load(open('texts.pkl', 'rb'))
        classes = pickle.load(open('labels.pkl', 'rb'))
        model = load_model('model.h5')
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        raise e
    return intents, words, classes, model

intents, words, classes, model = load_data()

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence, removing stopwords."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in stop_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """Convert a sentence into a bag of words representation."""
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words))
    word_set = set(sentence_words)  # Convert to set to remove duplicates
    for i, w in enumerate(words):
        if w in word_set:
            bag[i] = 1
            if show_details:
                logging.debug(f"Found in bag: {w}")
    return bag

def predict_class(sentence, model):
    """Predict the intent of the sentence using the trained model."""
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    results = [(i, r) for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i], "probability": str(prob)} for i, prob in results]

def getResponse(ints, intents_json):
    """Get the response for a predicted intent."""
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(msg):
    """Generate a response from the chatbot."""
    return getResponse(predict_class(msg, model), intents)

# Create a Flask Blueprint for the chatbot routes
chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route("/get")
def get_bot_response():
    """Handle the user's message and return a bot response."""
    userText = request.args.get('msg')
    logging.debug(f"Received message: {userText}")
    bot_response_text = chatbot_response(userText)
    return bot_response_text

# Initialize the Flask app
app = Flask(__name__)
app.static_folder = 'static'
app.register_blueprint(chatbot_bp)

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

if __name__ == "__main__":
    # Run the app
    app.run()
