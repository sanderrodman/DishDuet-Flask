from ast import arg
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import numpy as np
import pandas as pd

from scipy.sparse.linalg import svds

import scipy.sparse

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

try: 
    from credentials import LOCAL_MYSQL_USER_PASSWORD
except: 
    LOCAL_MYSQL_USER_PASSWORD = "cs4300food"


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = LOCAL_MYSQL_USER_PASSWORD
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "dishduet"

mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER,LOCAL_MYSQL_USER_PASSWORD,LOCAL_MYSQL_PORT,LOCAL_MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

allergies_dic = {
    "nut": ["almond", "almond butter", "beechnut", "butternut", "peanut",
             "cashew", "chestnut", "hazelnut", "macadamia", "pecan",
             "pine", "pistachio", "praline", "walnut", "brazil"],
    "vegetarian": ["ham", "beef", "chicken", "turkey", "duck", "squid", "jerky",
                   "bacon", "pork", "fish", "lamb", "lobster", "snail", "meatball",
                   "scallop", "frog", "snail", "oyster", "rabbit", "octopus",
                   "clam", "mussel", "goat", "fowl", "venison", "crab",
                   "sausage", "veal", "tuna", "salmon", "tilapia", "pepperoni"],
    "dairy": ["butter", "ghee", "casein", "cheese", "cream", "goat cheese",
              "milk", "yogurt", "custard", "sour cream", "ice cream",
              "whipped cream", "blue cheese", "cottage cheese", "whole milk", 
              "half and half", "whipped cream", "skim milk"],
    "egg": ["egg", "quail egg", "duck egg"],
    "gluten": ["cereal", "rye", "barley", "oats", "flour", "pasta",
               "bagel", "beer", "wheat"],
    "shellfish": ["snail", "lobster", "clam", "squid", "oyster",
                  "scallop", "crayfish", "prawn", "octopus",
                  "shrimp", "crab", "crawfish"],
}
allergies_dic["vegan"] = allergies_dic["vegetarian"] + allergies_dic["egg"] + allergies_dic["dairy"]

keys = ["time","dishname","cooktime","preptime","totaltime","detail","recipecategory","keywords",\
            "ingredientparts","aggregatedrating","reviewcount","calories","instructions","images","subpage", "score", "dimension", "dimension_score"]

df = pd.read_sql(f"""SELECT * FROM recipes""", mysql_engine.lease_connection().connection) # type: ignore

df["ingredientparts_str"] = df["ingredientparts"].apply(lambda x : str(x).replace(',', ''))
df["keywords_str"] = df["keywords"].apply(lambda x : str(x).replace(',', '')) + " " + df["recipecategory"]

df["ingredientparts"] = df["ingredientparts"].apply(lambda x : str(x).split(', '))

df["combined"] = df["dishname"] + " " + df["keywords_str"] + " " + df["ingredientparts_str"] + " " + df["detail"] + " " + df["instructions"]
df["combined"] = df["combined"].astype(str)

vocab_vectorizer = CountVectorizer(stop_words = 'english', max_df = .9, min_df = 3)
vocab_vectorizer.fit_transform(df["combined"].tolist())
vocabulary = vocab_vectorizer.vocabulary_
inv_vocabulary = {v: k for k, v in vocabulary.items()}


# svd stuff
vectorizer = TfidfVectorizer(stop_words = 'english', vocabulary=vocabulary)

dt_matrix = 0.9 * vectorizer.fit_transform(df["combined"]) + 0.1 * vectorizer.fit_transform(df["dishname"])

docs_compressed, s, words_compressed = svds(dt_matrix, k=1000)

words_compressed = np.array(words_compressed, dtype=float).T

docs_compressed = np.array(docs_compressed, dtype=float)

words_compressed_normed = np.array(normalize(words_compressed), dtype=float)

docs_compressed_normed = np.array(normalize(docs_compressed), dtype=float)


rating_scores = df["aggregatedrating"] * np.sqrt(df["reviewcount"]) + 69
rating_scores = np.array(np.log10(rating_scores), dtype=float)

def svd_search(query, unwanted, allergies, time): # runs on search

    df_return = df.drop(columns = ["ingredientparts_str", "keywords_str", "combined"])

    if len(allergies) != 0 or time < 60:
        df_return = filter(df_return, allergies, time)

    index = df_return.index.to_numpy()

    similarity, query_vec = cossim_sim(query, unwanted, index)
    
    scores = np.array(similarity * rating_scores[index], dtype=float)
    args = np.argsort(-scores.flatten())

    df_return = df_return.iloc[args].iloc[:30].reset_index(drop=True)

    dimensions, dimension_scores = dimension_and_score(args[:30], docs_compressed_normed, query_vec)
  
    df_return["score"] = pd.Series(np.round(100 * similarity[args], 2), dtype=float)
    df_return["dimension"] = pd.Series(dimensions, dtype=str)
    df_return["dimension_score"] = pd.Series(np.round(100 * dimension_scores, 2), dtype=float)
    print(df_return.head())
    results = df_return.values.tolist()

    return json.dumps([dict(zip(keys,i)) for i in results])


# cosine similarity
def cossim_sim(query, unwanted, index):

    query_tfidf = normalize(vectorizer.transform([query]) - 1.5 * vectorizer.transform([unwanted])) # type: ignore

    query_vec = np.array(np.dot(query_tfidf.toarray(), words_compressed_normed)).squeeze() # type: ignore

    scores = docs_compressed_normed[index].dot(query_vec)
    
    return (np.array(scores, dtype=float), query_vec)


def dimension_and_score(top_args, docs_compressed_normed, query_vec):
    dimensions = []
    dimension_scores = []

    for doc in docs_compressed_normed[top_args] * query_vec:
        dimension_col = words_compressed_normed[:, doc.argmax()].squeeze()
        top2_args = np.argpartition(dimension_col, -2)[-2:]
        dimensions.append(inv_vocabulary[top2_args[0]] + " or " + inv_vocabulary[top2_args[1]])
        dimension_scores.append(np.average(dimension_col[top2_args]))

    return (dimensions, np.array(dimension_scores, dtype=float))


def filter(df_filter, allergies, time): # boolean not search

    df_filter = pd.DataFrame(df_filter)
 
    if len(allergies) != 0:

        for allergie_category in allergies:

            indices = df_filter["ingredientparts"].apply(lambda ingredients : filterIngredients(allergie_category, ingredients))

            df_filter = df_filter[indices]
                                  
    if time < 60 and time > 0:

        indices = df_filter["time"] <= time

        df_filter = df_filter[indices]

    return df_filter


def filterIngredients(allergie_category, ingredients):
    
    for ingredient in ingredients:
        
        for allergie_str in allergies_dic[allergie_category]:

            if (allergie_str in ingredient.lower()):

                return False
    
    return True


@app.route("/")
def home():
    return render_template('base.html',title="sample html")


@app.route("/recipes")
def search():

    dishname = str(request.args.get("q")).lower().strip()

    unwanted = str(request.args.get("unwanted")).lower().strip()

    allergies = str(request.args.get("allergies")).strip().split()

    time = int(request.args.get("time")) # type: ignore
    
    return svd_search(dishname, unwanted, allergies, time)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
