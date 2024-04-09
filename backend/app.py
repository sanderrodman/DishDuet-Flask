import json
import os
from re import L
from unittest import result
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import numpy as np
import pandas as pd
import sqlalchemy as db

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
                  "shrimp", "crab"]
}

allergies_dic["vegan"] = allergies_dic["vegetarian"] + allergies_dic["egg"] + allergies_dic["dairy"]

# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
def sql_search(name, unwanted, allergies, time):

    query_sql = f"""SELECT * FROM recipes WHERE LOWER( dishname ) LIKE '%%%%{name}%%%%'"""

    # keys = ["id","dishname","cooktime","preptime","totaltime","detail","recipecategory","keywords",\
    #         "ingredientquantities","ingredientparts","aggregatedrating","reviewcount","calories",\
    #             "fat","saturdatedfat","cholesterol","sodium","carbs","fiber","sugar","protein",\
    #                 "instructions","images"]
    
    keys = ["dishname","time","cooktime","preptime","totaltime","detail","recipecategory","keywords",\
            "ingredientparts","aggregatedrating","reviewcount","calories",\
                "fat","sodium","carbs","fiber","sugar","protein",\
                    "instructions","images"]

    df = pd.read_sql(query_sql, mysql_engine.lease_connection().connection)
        
    df["ingredientparts"] = df["ingredientparts"].apply(lambda x : str(x).split(', '))
    df["keywords"] = df["keywords"].apply(lambda x : str(x).split(', '))

    df = filter(df, unwanted, allergies, time)

    return json.dumps([dict(zip(keys,i)) for i in df])


def filter(df, unwanted, allergies, time):
    if len(unwanted) != 0:
        df = df[df["ingredientparts"].apply(lambda x : ingredient_distance(unwanted, x))]

    if len(allergies) !=0:
        for allergie in allergies:
            df = df[df["ingredientparts"].apply(lambda x : \
                ingredient_distance(allergies_dic[allergie], x))]

    if time != 0:
        df = df[df["time"] <= time]

    return df.values[:15]


def ingredient_distance(sources, targets):

    for source in sources:
        for target in targets:
            print(source, target)
            if distance(source, target) < 3:
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

    dishname = request.args.get("dishname").lower().strip()

    unwanted = request.args.get("unwanted").lower().strip()

    unwanted = unwanted.split(",") if "," in unwanted else unwanted.split()

    allergies = request.args.get("allergies").strip().split()

    time = int(request.args.get("time"))

    return sql_search(dishname, unwanted, allergies, time)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
