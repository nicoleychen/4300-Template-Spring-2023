# from IPython.core.display import HTML
# from nltk.tokenize import TreebankWordTokenizer
# import numpy as np
# import time
# import string
# import math
# from collections import Counter
# from collections import defaultdict
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

from dotenv import load_dotenv

load_dotenv()

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_USER_PASSWORD = os.getenv('MYSQL_USER_PASSWORD')
MYSQL_PORT = 3306
MYSQL_DATABASE = "kardashiandb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this


def sql_search(episode):
    query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
    keys = ["id", "title", "descr"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys, i)) for i in data])


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text)


# app.run(debug=True)


# TODO: figure out if json is correct in importing from sql datbase
# TODO: figure out how to connect to sql and import sql database --> i think data is on my personal database connection
#      so changing the dtabase connection ports in app.py to my root would work for me, but it doesn't work for you guys i don't think

# TODO: maybe for now use csv file format initially?
# with open("perfume-data.json") as f:
#     perfumes = json.load(f)


# TODO: split db into data of just perfume name with notes


# def build_inverted_index(database):
#     """ Builds an inverted index from the perfume name and notes.

#     Arguments
#     =========

#     database: list of dicts.
#         Each perfume in this list already has a 'notes'
#         field that contains the tokenized notes.

#     Returns
#     =======

#     inverted_index: dict
#         For each note, the index contains
#         a list of that stores all the perfume_id with that note.
#         inverted_index[note] = [p1, p2, p3]

#     """
#     res = {}
#     for i in range(len(database)):
#         id = database[i]
#         notes = database['perfume_notes']
#         notes_set = set(notes)
#         for note in notes_set:
#             if note not in res:
#                 res[note] = []
#             res[note].append(i)
#     return res
