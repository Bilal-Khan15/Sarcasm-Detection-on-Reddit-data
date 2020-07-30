from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle


# Flask App
app = Flask(__name__)
CORS(app)


def train():
    data=pd.read_csv('reddit_training.csv')
    data.drop(["index","author","created_utc","subreddit_id","link_id","parent_id","id","subreddit"], axis=1, inplace=True)
    Y=data.sarcasm_tag
    x_train,x_test,y_train,y_test=train_test_split(data,Y,test_size=0.2,random_state=7)
    tfidf_vectorizer  = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train['body']) 
    #tfidf_test  = tfidf_vectorizer.transform(x_test['body'])
    
    model=PassiveAggressiveClassifier(max_iter=50)
    model.fit(tfidf_train,y_train)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    pickle.dump(tfidf_vectorizer, open("tfidf_model.sav", 'wb'))
    
def predict(text):
    tf_idf = "tfidf_model.sav"
    tfidf_vectorizer = pickle.load(open(tf_idf, 'rb'))
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    text = [text]
    text = tfidf_vectorizer.transform(text)
    y_pred=model.predict(text)
    return y_pred

train()

@cross_origin
@app.route('/detect', methods=['POST'])
def sarcasm_detector():
    count = dict(request.form)
    count = count['type']

    result = predict(count)
    if(result[0] == 'no'): count = "It's not Sarcastic !"
    else: count = "It's Sarcastic !"

    # Return Response
    return jsonify({"result": count})


if __name__ == '__main__':
    app.run(debug=False)