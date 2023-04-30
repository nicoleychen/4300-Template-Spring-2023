# from IPython.core.display import HTML
# from nltk.tokenize import TreebankWordTokenizer
import numpy as np
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
from fuzzywuzzy import fuzz

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
MYSQL_DATABASE = "findmyfragrance_db"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

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


# TODO: add def
# app.run(debug=True)

# JJ: function that opens json file
def load_perfume_data():
    f = open('perfume_data_combined.json')
    data = json.load(f)
    print("JSON succesfully loaded!")
    f.close()
    return data


perfume_json = load_perfume_data()

# search autocomplete 
@app.route("/suggestion/perf")
def suggest_perf():
    text = request.args.get("name")
    print(perfume_name_suggest(text))
    return perfume_name_suggest(text)

# checks if query is in dataset or not


@app.route("/testing")
def testing_search():
    query = request.args.get("name")
    print("test query: " + str(query))
    for _, name in perfume_json["name"].items():
        if name == query:
            return json.dumps(True)
    return json.dumps(False)


@app.route("/self")
def get_query_info():
    name = request.args.get("name")

    exists = False
    for i, perf_name in perfume_json["name"].items():
        if perf_name == name:
            idx = i
            exists = True

    if not exists:
        return json.dumps(False)

    info = {}
    info["img"] = perfume_json["image"][idx]
    info["gender"] = perfume_json["for_gender"][idx]
    info["name"] = perfume_json["name"][idx]
    info["brand"] = perfume_json["company"][idx]
    info["rating"] = perfume_json["rating"][idx]
    info["gender"] = perfume_json["for_gender"][idx]
    info["topnote"] = perfume_json["top notes"][idx]
    info["middlenote"] = perfume_json["middle notes"][idx]
    info["bottomnote"] = perfume_json["base notes"][idx]
    info["desc"] = perfume_json["description"][idx]

    return json.dumps(info)


# this is main route combining query and gender preference
@app.route("/similar")
def similar_search():
    result = []
    name = request.args.get("name")
    gender_pref = request.args.get("gender_pref")
    print("name: " + name)
    print("gender:" + gender_pref)

    exists = False
    for _, perf_name in perfume_json["name"].items():
        if perf_name == name:
            exists = True
    if not exists:
        return json.dumps(result)
    # check if name is in perfume json, if not return "sorry pick another name"
    # Rest of algorithm goes here
    # 1. gender filter
    all_ids = list(perfume_json["name"].keys())
    gendered_ids = gender_filter(perfume_json, all_ids, gender_pref)
    # 2. rating threshold filter
    rated_ids = rating_threshold_filter(perfume_json, gendered_ids)

    # 3. jaccard sim filter
    num_perfumes = len(rated_ids)
    perf_data = perfume_json_to_all_notes(perfume_json, rated_ids)

    jaccard = build_perf_sims_jac(num_perfumes, perf_data)
    perfume_ind_to_id = perfume_index_to_id(perf_data)
    ranked = get_ranked_perfumes(name, jaccard, perfume_ind_to_id, perf_data)
    result = results(ranked, perfume_json)

    return json.dumps(result)


"""
CODE FROM https://github.com/Y1chenYao/thank-u-next-cornell-prof-recommender/blob/master/backend/app.py
note: functions for edit distance in dropdowns
"""
def perfume_name_suggest(input_perf):
    perf_scores = {}
    perf_name_list = get_perfume_names(perfume_json)
    # perf_list is json file
    for perf in perf_name_list:
        score = fuzz.partial_ratio(input_perf.lower(), perf.lower())
        perf_scores[perf] = score
    sorted_perfs = sorted(perf_scores.items(), key=lambda x:x[1], reverse=True)[:5]
    return json.dumps([perf[0] for perf in sorted_perfs])


# NC: gets input gender preference from frontend and returns it
# @app.route("/gender_pref")
# def gender_search():
#     query = request.args.get("gender")
#     print("gender query: " + str(query))
#     pref = ""
#     # men
#     if query == "male":
#         pref = "for men"
#     # women, idk why it's on but i'm j rolling w it
#     elif query == "on":
#         pref = "for women"
#     # no pref
#     else:
#         pref = "for women and men"
#     gender = pref
#     return json.dumps(pref)

# NC: uses gender_search to filter by input gender preference. 
# Takes in perfume_data JSON and a list of ids that correspond to perfumes.
# Returns a list of filtered ids that correspond to the input gender preference.


def gender_filter(perfume_data, perfume_ids, gender_filter):
    # set query to result of gender search somehow, this doesn't work
    # query = gender_search()
    res = []
    for id in perfume_ids:
        if perfume_data["for_gender"][str(id)] != gender_filter:
            res.append(id)
    return res


# NC: takes in perfume_data JSON and a list of ids that correspond to perfumes.
# Returns a filtered list of ids that only correspond to those with above 3 star ratings.
def rating_threshold_filter(perfume_data, perfume_ids, threshold=3.0):
    res = []
    for id in perfume_ids:
        if (perfume_data["rating"][str(id)]) != "NA" and float(perfume_data["rating"][str(id)]) > threshold:
            res.append(id)
    return res

# *ONLY FOR IDS IN ids* loop through all the top middle bottom notes, returns a list of dictionaries that represent {'id' :perfume id, 'notes' : list of notes}
# THIS IS THE FILTERED PERFUME DATA TO USE


def perfume_json_to_all_notes(perfume_json, ids):
    top_note_json = perfume_json['top notes']
    middle_note_json = perfume_json['middle notes']
    base_note_json = perfume_json['base notes']
    res = []
    for id in top_note_json:
        if id in ids:
            perf = {}
            perf['id'] = id
            perf['name'] = perfume_json["name"][id]
            perf['notes'] = top_note_json[id] + \
                middle_note_json[id] + base_note_json[id]
            res.append(perf)
    return res


def get_perfume_names(perf_json):
    # gets all perfume names
    names = set(perf_json['name'].values())
    return names


def perfume_id_to_index(filtered_perf):
    # Builds dictionary where keys are the perfume id, and values are the index
    # takes in filtered perfume
    res = {}
    for i in range(len(filtered_perf)):
        perfume = filtered_perf[i]
        res[perfume['id']] = i
    return res


def perfume_index_to_id(filtered_perf):
    # Builds dictionary where keys are the perfume id numbers, and values are the index
    # takes in filtered perfume
    res = {}
    temp = perfume_id_to_index(filtered_perf)
    for k, v in temp.items():
        res[v] = k
    return res


def perfume_id_to_name(perf_json):
    # Dictionary {id : name}
    return perf_json['name']


def perfume_name_to_id(perf_json):
    # Dictionary {name : id}
    res = {}
    for k, v in perf_json['name'].items():
        res[v] = k
    return res


def perfume_name_to_index(filtered_perf):
    # Dictionary {name : ind}
    res = {}
    for i in range(len(filtered_perf)):
        res[filtered_perf[i]['name']] = i
    return res


def perfume_index_to_name(filtered_perf):
    # Dictionary {ind : name}
    res = {}
    for i in range(len(filtered_perf)):
        res[i] = filtered_perf[i]['name']
    return res

# get query perfume


def check_query(input_query, perf_json):
    # query = input_query.lower()
    perfume_names = get_perfume_names(perf_json)
    if input_query in perfume_names:
        return input_query
    return "Sorry, no results found. Check your spelling or try a different perfume."


def build_inverted_index(filtered_pref):
    """ Builds an inverted index from the perfume name and notes.
    Arguments
    =========
    database: list of dicts.
        Each perfume in this list already has a 'notes'
        field that contains the tokenized notes.
    Returns
    =======
    inverted_index: dict
        For each note, the index contains
        a list of that stores all the perfume_id with that note.
        inverted_index[note] = [p1, p2, p3]
    """
    res = {}
    for i in range(len(database)):
        id = database[i]
        notes = database['notes']
        notes_set = set(notes)
        for note in notes_set:
            if note not in res:
                res[note] = []
            res[note].append(i)
    return res


def build_perf_sims_jac(n_perf, input_data):
    """Returns a perf_sims_jac matrix of size (perf_movies,perf_movies) where for (i,j) :
        [i,j] should be the jaccard similarity between the category sets (notes) for perfumes i and j
        such that perf_sims_jac[i,j] = perf_sims_jac[j,i].
    Params: {n_perf: Integer, the number of perfumes,
            input_data: List<Dictionary>, a list of dictionaries that represent perfume id : list of all notes}
    Returns: Numpy Array
    """
    perf_sims = np.zeros((n_perf, n_perf))
    for i in range(n_perf):
        for j in range(n_perf):
            if i == j:
                perf_sims[i][j] = 1.0
            else:
                category_1 = set(input_data[i]["notes"])
                category_2 = set(input_data[j]["notes"])
                intersect = len(category_1.intersection(category_2))
                union = len(category_1.union(category_2))
                if union == 0:
                    perf_sims[i][j] = 0
                else:
                    perf_sims[i][j] = intersect/union
    return perf_sims

# rank all perfumes, and return top 3


def get_ranked_perfumes(perfume, matrix, perf_index_to_id, filtered_perf):
    """
    Return top 5 of sorted rankings (most to least similar) of perfumes as
    a list of two-element tuples, where the first element is the
    perfume name and the second element is the similarity score
    Params: {perfume: String,
             matrix: np.ndarray (the output of build_perf_sims_jac)
             perf_index_to_name: dict}
    Returns: List<Tuple> (id, score)
    """
    # Get movie index from movie name
    perf_idx = perfume_name_to_index(filtered_perf)[perfume]
    # Get list of similarity scores for movie
    score_lst = matrix[perf_idx]
    perf_score_lst = [(perf_index_to_id[i], s)
                      for i, s in enumerate(score_lst)]
    # Do not account for movie itself in ranking
    perf_score_lst = perf_score_lst[:perf_idx] + perf_score_lst[perf_idx+1:]
    # Sort rankings by score
    perf_score_lst = sorted(perf_score_lst, key=lambda x: -x[1])
    return perf_score_lst[:5]

# get the necessary information


def results(top_5, perf_json):
    """
        Take in list of top 5 perfumes ids and get the corresponding info
        input_dict: list of dictionaries for each perfume - perf_dict

        Returns: 
    """
    final = []
    for i in range(len(top_5)):
        info = {}
        info["img"] = perf_json["image"][top_5[i][0]]
        info["gender"] = perf_json["for_gender"][top_5[i][0]]
        info["name"] = perf_json["name"][top_5[i][0]]
        info["brand"] = perf_json["company"][top_5[i][0]]
        info["rating"] = perf_json["rating"][top_5[i][0]]
        info["gender"] = perf_json["for_gender"][top_5[i][0]]
        info["topnote"] = perf_json["top notes"][top_5[i][0]]
        info["middlenote"] = perf_json["middle notes"][top_5[i][0]]
        info["bottomnote"] = perf_json["base notes"][top_5[i][0]]
        info["desc"] = perf_json["description"][top_5[i][0]]
        final.append(info)
    return final

#     final = []
#     perf_dict = {}
# inner_dict = {}
#     for row in database:
#         inner_dict["brand"] = row["Brand"]
#         inner_dict["description"] = row["Description"]
#         inner_dict["notes"] = row["Notes"]
# perf_dict[row["Name"]] = inner_dict
#     final.append(perf_dict)
# create perfume name to index, index to name ??
# perfume_id_to_index = {perfume_id:index for index, perfume_id in enumerate([d['perfume_id'] for d in data])}
# perfume_name_to_id = {name:pid for name, pid in zip([d['Name'] for d in data],
#                                                      [d['perfume_id'] for d in data])}
# perfume_id_to_name = {v:k for k,v in perfume_name_to_id.items()}
# perfume_name_to_index = {name:perfume_id_to_index[perfume_name_to_id[name]] for name in [d['perfume_name'] for d in data]}
# perfume_index_to_name = {v:k for k,v in perfume_name_to_index.items()}
# perfume_names = [name for name in [d['perfume_name'] for d in data]]
# get query perfume
# def check_query(input_query)
# query = input_query.lower()
# if query in perfume_names:
#     return query
# else:
#     return "Sorry, no results found".
# query_perfume = check_query(query)
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
# create jaccard similarity matrix
# def build_perf_sims_jac(n_perf, input_data):
#     """Returns a perf_sims_jac matrix of size (perf_movies,perf_movies) where for (i,j) :
#         [i,j] should be the jaccard similarity between the category sets (notes) for perfumes i and j
#         such that perf_sims_jac[i,j] = perf_sims_jac[j,i].
#     Params: {n_perf: Integer, the number of perfumes,
#             input_data: List<Dictionary>, a list of dictionaries where each dictionary
#                      represents the perfume_data including the perfume and the metadata of each perfume}
#     Returns: Numpy Array
#     """
#     perf_sims = np.zeros((n_perf, n_perf))
#     for i in range(n_perf):
#         for j in range(n_perf):
#             if i==j:
#                 perf_sims[i][j] = 1.0
#             else:
#                 category_1 = set(input_data[i]["notes"])
#                 category_2 = set(input_data[j]["notes"])
#                 intersect = len(category_1.intersection(category_2))
#                 union = len(category_1.union(category_2))
#                 perf_sims[i][j] = intersect/union
#     return perf_sims
# #rank all movies, and return top 3
# def get_ranked_movies(perfume, matrix):
#     """
#     Return sorted rankings (most to least similar) of perfumes as
#     a list of two-element tuples, where the first element is the
#     perfume name and the second element is the similarity score
#     Params: {perfume: String,
#              matrix: np.ndarray}
#     Returns: List<Tuple>
#     """
#     # Get movie index from movie name
#     perf_idx = perfume_name_to_index[perfume]
#     # Get list of similarity scores for movie
#     score_lst = matrix[perf_idx]
#     perf_score_lst = [(perf_index_to_name[i], s) for i,s in enumerate(score_lst)]
#     # Do not account for movie itself in ranking
#     perf_score_lst = perf_score_lst[:perf_idx] + perf_score_lst[perf_idx+1:]
#     # Sort rankings by score
#     perf_score_lst = sorted(perf_score_lst, key=lambda x: -x[1])
#     print("Top {} most similar movies to {} [{}]".format(k, 'star wars'))
#     print("======")
#     for (mov, score) in perf_score_lst[:3]:
#         print("%.3f %s" % (score, mov))
# return top_3 = perf_score_lst[:3]
# get the necessary information
# def results(top_3, input_dict):
#     """
#         Take in list of top 3 movies and get the correspoiding info
# input_dict: list of dictionaries for each perfume - perf_dict
#     """
# dicts = []
#     for i in top 3:
# dicts.append(input_dict[i])
# return dicts
