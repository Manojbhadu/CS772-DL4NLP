import tkinter as tk
import nltk
import gensim
import gensim.downloader
from nltk.corpus import brown
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd 
import pickle
import numpy as np
from keras.models import *

def preprocess_test(test_reviews):
  reviews = test_reviews
  reviews = convert_to_lower(reviews)
  reviews = perform_tokenization(reviews)
  reviews = remove_punctuation(reviews)
  return reviews

def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    lower_text = text.copy()
    for i in range(len(text)):
        lower_text[i] = text[i].lower()
    return lower_text

def remove_punctuation(text):
    #stop_words = set(stopwords.words('english'))
    without_punctuation_text  = text.copy()
    for i in range(len(text)):
        without_punctuation_text[i] = [w for w in text[i] if w.isalpha()]
    return without_punctuation_text

def perform_tokenization(text):
    tokenize_text = text.copy()
    for i in range(len(text)):
        tokenize_text[i] = nltk.word_tokenize(text[i])
    return tokenize_text


def get_dicts(train):
    reviews = train["reviews"]
    reviews = convert_to_lower(reviews)
    reviews = perform_tokenization(reviews)
    reviews = remove_punctuation(reviews)
    return reviews



def predict(model, test_reviews):
        y_pred = model.predict(test_reviews)
        pred1 = []
        for i in range(len(y_pred)):
            pred1.append(np.argmax(y_pred[i])+1)
        return pred1

    
train = pd.read_csv("train.csv")
review = get_dicts(train)
review = review.to_list()
max_length=50


tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review) 
sequences = tokenizer_obj.texts_to_sequences(review)
word_index = tokenizer_obj.word_index
review_pad = pad_sequences(sequences, maxlen=max_length) 

model = load_model("./bestModel.h5")

def function():
    result = T.get("1.0", "end")
    if result=="\n":
        tk.messagebox.showinfo('Message', "Enter Sentence")
    else :
        result = result.rstrip("\n")
        r = preprocess_test([result])
        sequences_test = tokenizer_obj.texts_to_sequences(r)
        test_review_pad = pad_sequences(sequences_test, maxlen=max_length) 
        answer = predict(model,test_review_pad)
        tk.messagebox.showinfo('Message', "The predicted sentiment of the sentence is " + str(answer[0]))

window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry("500x200")

l = tk.Label(window, text = "Enter Sentence")
l.config(font =("Courier", 14))
T = tk.Text(window, height = 5, width = 52)



button = tk.Button(window, text='Predict', width=10, command=function)

l.pack()
T.pack()

button.pack()
window.mainloop()

