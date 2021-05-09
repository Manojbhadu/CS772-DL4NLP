import tkinter as tk
import tensorflow as tf 
from transformers import BertTokenizer
import numpy as np 
from transformers import TFAutoModel
from tkinter import messagebox

bert = TFAutoModel.from_pretrained('bert-base-cased')  #, output_hidden_states=False

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

model = tf.keras.models.load_model("./model.h5")


def prep_data(text):
    tokens = tokenizer.encode_plus(text, max_length=128,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_tensors='tf')
    # tokenizer returns int32 tensors, we need to return float64, so we use tf.cast
    return {'input_ids': tf.cast(tokens['input_ids'], tf.float64),
            'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}

def function():
    result = T.get("1.0", "end")
    if result=="\n":
        messagebox.showinfo('Message', "Enter Sentence")
    else :
        result = result.rstrip("\n")
        probs = model.predict(prep_data(result))[0]
        answer = np.argmax(probs) + 1
        messagebox.showinfo('Message', "The predicted sentiment of the sentence is " + str(answer))

window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry("500x500")

l = tk.Label(window, text = "Enter Sentence")
l.config(font =("Courier", 20))
T = tk.Text(window, height = 20, width = 60)
button = tk.Button(window, text='Predict', width=20, command=function)
l.pack()
T.pack()
button.pack()
window.mainloop()

