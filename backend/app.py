import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

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
              "half and half", "whipped cream"],
    "egg": ["egg", "quail egg", "duck egg"],
    "gluten": ["cereal", "rye", "barley", "oats", "flour", "pasta",
               "bagel", "beer", "wheat"],
    "shellfish": ["snail", "lobster", "clam", "squid", "oyster",
                  "scallop", "crayfish", "prawn", "octopus",
                  "shrimp", "crab", "crawfish"],
}
allergies_dic["vegan"] = allergies_dic["vegetarian"] + allergies_dic["egg"] + allergies_dic["dairy"]

keys = ["dishname","time","cooktime","preptime","totaltime","detail","recipecategory","keywords",\
            "ingredientparts","aggregatedrating","reviewcount","calories", "instructions","images"]

df = pd.read_sql(f"""SELECT * FROM recipes""", mysql_engine.lease_connection().connection)

df["ingredientparts_str"] = df["ingredientparts"].apply(lambda x : str(x).replace(',', ''))
df["keywords_str"] = df["keywords"].apply(lambda x : str(x).replace(',', '')) #could be done better

df["ingredientparts"] = df["ingredientparts"].apply(lambda x : str(x).split(', '))
df["keywords"] = df["keywords"].apply(lambda x : str(x).split(', '))

def svd_search(query, unwanted, allergies, time, r=15):

    sum = cossim_sum(query) - cossim_sum(unwanted)

    args = np.argsort(-sum)

    results = filter(df.iloc[args], allergies, time).iloc[:r].values.tolist()

    return json.dumps([dict(zip(keys,i)) for i in results])

# cosine similarity
def cossim_sum(query):

    sum = 0
    for svd, weight in field_to_svds_and_weights:

        vectorizer, words_compressed, docs_compressed_normed = svd

        query_tfidf = vectorizer.transform([query]).toarray()

        query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

        sum += weight * docs_compressed_normed.dot(query_vec)
    
    return sum


def svd_maker(field):
        
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .9, min_df = 2)
    td_matrix = vectorizer.fit_transform(df[field].to_list())

    # word_to_index = vectorizer.vocabulary_
    # index_to_word = {i:t for t,i in word_to_index.items()}

    docs_compressed, s, words_compressed = svds(td_matrix, k=100)
    words_compressed = words_compressed.transpose()

    # words_compressed_normed = normalize(words_compressed, axis = 1)
    docs_compressed_normed = normalize(docs_compressed, axis = 1)

    td_matrix_np = td_matrix.transpose().toarray()
    td_matrix_np = normalize(td_matrix_np)

    return (vectorizer, words_compressed, docs_compressed_normed)

field_to_svds_and_weights = ( # must be under svd_maker()
    (svd_maker("dishname"), 4),
    (svd_maker("ingredientparts_str"), 3),
    (svd_maker("keywords_str"), 3),
    (svd_maker("detail"), 3),
)


def filter(df, allergies, time): # boolean not search ??
    # if len(unwanted) != 0:
    #     df = df[df["ingredientparts"].apply(lambda x : ingredient_distance(unwanted, x))]

    if len(allergies) !=0:
        for allergie in allergies:
            df = df[df["ingredientparts"].apply(lambda x : \
                ingredient_distance(allergies_dic[allergie], x))]

    if time != 0:
        df = df[df["time"] <= time]

    return df


def ingredient_distance(sources, targets):

    for source in sources:
        for target in targets:
            if distance(source, target) < 3: # 3 is edit threshold
                return False
    return True


def distance(source, target):
    
    d = np.zeros((len(source), len(target)))

    for i in range(d.shape[0]):
        d[i,0] = i
    
    for j in range(d.shape[1]):
        d[0,j] = j

    for i in range(d.shape[0]):

        for j in range(d.shape[1]):

            subcost = 0 if source[i] == target[j] else 1

            d[i, j] = np.min(np.array([d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + subcost]))

    return d[i, j]


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/recipes")
def search():

    dishname = str(request.args.get("dishname")).lower().strip()

    # unwanted = str(request.args.get("unwanted")).lower().strip() old unwanted, save for now
    # unwanted = unwanted.split(",") if "," in unwanted else unwanted.split()

    unwanted = str(request.args.get("unwanted")).lower().strip()

    allergies = str(request.args.get("allergies")).strip().split()

    time = int(request.args.get("time"))
    
    return svd_search(dishname, unwanted, allergies, time)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
