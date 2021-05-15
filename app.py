import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KDTree


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #Reading the dataset
    person = pd.read_csv('famous_people.csv')

    #Printing first five rows of the dataset
    #print(person.head())

    #Counting the frequency of occurance of each word
    count_vector = CountVectorizer()
    train_counts = count_vector.fit_transform(person.Text)

    #Using tf-idf to reduce the weight of common words
    tfidf_transform = TfidfTransformer()
    train_tfidf = tfidf_transform.fit_transform(train_counts)
    a = np.array(train_tfidf.toarray())

    #obtaining the KDTree
    kdtree = KDTree(a ,leaf_size=3)

    #take the name of the personality as input
    person_name=request.form.get('person')

    #Using KDTree to get K articles similar to the given name
    person['tfidf']=list(train_tfidf.toarray())
    distance, idx = kdtree.query(person['tfidf'][person['Name']== person_name].tolist(), k=3)
    for i, value in list(enumerate(idx[0])):
	    output = output + "Name : {}".format(person['Name'][value]) +" "
	    "Distance : {}".format(distance[0][i]) + " "
	    "URI : {}".format(person['URI'][value]) +"\n"


    return render_template('index.html',prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)