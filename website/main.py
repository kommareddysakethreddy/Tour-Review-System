from mail import mailing
from flask import Flask, redirect, url_for, render_template, request
# importing libraries
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
model = load_model(
    r"C:\Users\saket\Desktop\xwebsite\ML\NLP\website\templates\model_rnn.h5")
f = open(r"C:\Users\saket\Desktop\xwebsite\ML\NLP\website\templates\tokenizer_data .pkl", 'rb')

# render mail function in another file


app = Flask(__name__)
app.secret_key = "Teams"


@app.route("/")
def home():
    return render_template("index.html", cont="User")


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        review = request.form['text']
        rating = 0
        confidence = 0

        data = pickle.load(f)
        tokenizer = data['tokenizer']
        # review = "The tour was amazing! The guide was very knowledgeable and the scenery was beautiful."
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(
            sequence, padding="post", maxlen=200, truncating="post")
        prediction = model.predict(padded_sequence)
        rating = np.argmax(prediction[0])+1
        confidence = prediction[0][rating-1]
        print("Prediction: ", prediction)
        print("Rating: ", rating)
        if confidence <= 0.5 or rating <= 3:
            # use mailing function in another python file
            # mailing("sakethreddy620@gmail.com")
            pass


@app.route('/submit', methods=['POST'])
def submit():
    # request.form['text']
    return 'You entered: {}'.format(request.form['text'])


if __name__ == "__main__":
    app.run(debug=True)
