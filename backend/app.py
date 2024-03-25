import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

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
    "nut": ["almond", "almond butter", "beechnut", "butternut",
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
def sql_search(name, unwanted, allergies):

    allergies_string = ""
    for allergie in allergies:
        for food in allergies_dic[allergie]:
            allergies_string += f"""AND LOWER( ingredientparts ) NOT LIKE '%%%%{food}%%%%' """

    unwanted_string = ""
    for unwant in unwanted:
        unwanted_string += f"""AND LOWER( ingredientparts ) NOT LIKE '%%%%{unwant}%%%%' """

    query_sql = f"""SELECT * FROM recipes WHERE LOWER( dishname ) LIKE '%%%%{name}%%%%' """\
        + unwanted_string + allergies_string + f"""limit 20"""

    keys = ["id","dishname","cooktime","preptime","totaltime","detail","recipecategory","keywords",\
            "ingredientquantities","ingredientparts","aggregatedrating","reviewcount","calories",\
                "fat","saturdatedfat","cholesterol","sodium","carbs","fiber","sugar","protein",\
                    "instructions","images"]
    
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys,i)) for i in data])

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/recipes")
def search():

    dishname = request.args.get("dishname").lower().strip()

    unwanted = request.args.get("unwanted").lower().strip()

    unwanted = unwanted.split(",") if "," in unwanted else unwanted.split()

    allergies = request.args.get("allergies").strip().split()

    return sql_search(dishname, unwanted, allergies)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
