
import tensorflow as tf
import pandas as pd
import numpy as np


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense  , InputLayer, Activation
from keras.optimizers import Adam
# ADD THE LIBRARIES YOU'LL NEED

'''
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
'''
MAX_LENGTH = 60

def get_dicts(train):
    reviews = train["reviews"]
    reviews = convert_to_lower(reviews)
    reviews = perform_tokenization(reviews)
    reviews = remove_punctuation(reviews)
    reviews = remove_stopwords(reviews)
    
    words = []
    for sent in reviews:
        for word in sent:
            words.append(word)
            
    #for phrase in test:
    #for word in phrase:
    #words.append(word)
            
    words.sort()
    words = set(words)
    word_to_index = {}
    word_to_index['<PAD>'] = 0
    word_to_index['<UNK>'] = 1
    for i, word in enumerate(words):
        word_to_index[word] = i  + 2
        
    #index_to_word = {index:word for (word, index) in word_to_index.items()}
    
    return word_to_index


def encode_data(text,word_to_index):

    # This function will be used to encode the reviews using a dictionary(created using corpus vocabulary) 
    
    # Example of encoding :"The food was fabulous but pricey" has a vocabulary of 4 words, each one has to be mapped to an integer like: 
    # {'The':1,'food':2,'was':3 'fabulous':4 'but':5 'pricey':6} this vocabulary has to be created for the entire corpus and then be used to 
    # encode the words into integers 

    # return encoded examples
    encoded_text = []
    for phrase in text:
        tokenized_format = []
        for word in phrase:
            try:
                index = word_to_index[word]
                tokenized_format.append(index)
            except KeyError:
                index = word_to_index['<UNK>']
                tokenized_format.append(index)
            
        encoded_text.append(tokenized_format)
    return encoded_text


def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    lower_text = text.copy()
    for i in range(len(text)):
        lower_text[i] = text[i].lower()
    return lower_text


def remove_punctuation(text):
    # return the reviews after removing punctuations
    without_punctuation_text  = text.copy()
    for i in range(len(text)):
        without_punctuation_text[i] = [w for w in text[i] if w.isalpha()]
    return without_punctuation_text

def remove_stopwords(text):
    # return the reviews after removing the stopwords
    stop_words = set(stopwords.words('english'))
    without_stopwords_text  = text.copy()
    for i in range(len(text)):
        without_stopwords_text[i] = [w for w in text[i] if not w in stop_words]
    return without_stopwords_text

def perform_tokenization(text):
    # return the reviews after performing tokenization
    tokenize_text = text.copy()
    for i in range(len(text)):
        tokenize_text[i] = nltk.word_tokenize(text[i])
    return tokenize_text

def perform_padding(data):
    # return the reviews after padding the reviews to maximum length
    data = pad_sequences(data,maxlen=MAX_LENGTH,padding='post')
    return data

def preprocess_data(data,word_to_index):
    # make all the following function calls on your data
    # EXAMPLE:->
        '''
        review = data["reviews"]
        review = convert_to_lower(review)
        review = remove_punctuation(review)
        review = remove_stopwords(review)
        review = perform_tokenization(review)
        review = encode_data(review)
        review = perform_padding(review)
        '''
    # return processed data
    reviews = data["reviews"]
    reviews = convert_to_lower(reviews)
    reviews = perform_tokenization(reviews)
    reviews = remove_punctuation(reviews)
    reviews = remove_stopwords(reviews)
    reviews = encode_data(reviews,word_to_index)
    reviews = perform_pedding(reviews)
    return reviews


def softmax_activation(x):
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)


class NeuralNet:

    def __init__(self, reviews, ratings):

        self.reviews = reviews
        self.ratings = ratings



    def build_nn(self):

        #add the input and output layer here; you can use either tensorflow or pytorch
        self.model = Sequential()
        MAX_LENGTH = 60
        self.model.add(InputLayer(input_shape=(MAX_LENGTH, )))
        self.model.add(Dense(5,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
        #self.model.summary()


    def train_nn(self,batch_size,epochs):
        # write the training loop here; you can use either tensorflow or pytorch
        # print validation accuracy
        y_train = tf.keras.utils.to_categorical(self.ratings,num_classes=5)
        self.history = self.model.fit(self.reviews, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    def predict(self, reviews):
        # return a list containing all the ratings predicted by the trained model
        y_pred = self.model.predict(reviews)
        pred1 = []
        for i in range(len(y_pred)):
            pred1.append(np.argmax(y_pred[i])+1)
        return pred1


# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    
    batch_size,epochs=64,10
    
    word_to_index = get_dicts(train_file)
    train_reviews=preprocess_data(train_data)
    test_reviews=preprocess_data(test_data)

    train_rating_list = train['ratings'].to_list()
    Y = [str(i-1) for i in train_rating_list]
    model=NeuralNet(train_reviews,Y)
    model.build_nn()
    model.train_nn(batch_size,epochs)

    return model.predict(test_reviews)