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
from sklearn.feature_extraction.text import TfidfVectorizer

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
    f = open('perfumes_combined_0_299.json')
    data = json.load(f)
    print("JSON succesfully loaded!")
    f.close()
    return data


perfume_json = load_perfume_data()


# creating tfidf matrix
# def load_sample_data():
#     f = open('perfume_combined_0_150.json')
#     data = json.load(f)
#     print("JSON succesfully loaded!")
#     f.close()
#     return data

def format_json(j):
    reviews = j["reviews"]
    perf_list = []
    for perfume_id in reviews:
        temp = reviews[perfume_id]
        review_total = ""
        for review_id in temp:
            review_total = review_total + " " + temp[review_id]
        perf_list.append({"review": review_total})

    return perf_list

# *ONLY FOR IDS IN ids* loop through all the top middle bottom notes,
# returns a list of dictionaries that represent {'id' :perfume id, 'notes' : list of notes}


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


def get_common_keywords(perf1, perf2):
    keywords = []
    perfid1 = name_to_index[perf1]
    perfid2 = name_to_index[perf2]
    vector1 = perfume_by_term[perfid1]
    vector2 = perfume_by_term[perfid2]
    diff = np.absolute(np.subtract(vector1, vector2))
    diff_sorted = np.argsort(diff)
    count = 0
    for word_id in diff_sorted:
        if count==10:
            break
        word = index_to_vocab[word_id]
        if not (word.isnumeric()) and diff[word_id]!=0:
            print(diff[word_id])
            keywords.append(word)
            count+=1
    return keywords

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    """Returns a TfidfVectorizer object with the above preprocessing properties.
        The term document matrix of the perfume reviews input_doc_mat[i][j] is the tfidf
        of the perfume i for the word j.

    Note: This function may log a deprecation warning. This is normal, and you
    can simply ignore it.

    Parameters
    ----------
    max_features : int
        Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer 
        constructer.
    stop_words : str
        Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer. 
    max_df : float
        Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer. 
    min_df : float
        Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer. 
    norm : str
        Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer. 

    Returns
    -------
    TfidfVectorizer
        A TfidfVectorizer object with the given parameters as its preprocessing properties.
    """
    v = TfidfVectorizer(max_features=max_features, stop_words=stop_words,
                        max_df=max_df, min_df=min_df, norm=norm)
    return v


# example_json = load_sample_data()
formatted_data = format_json(perfume_json)
# review_vec = build_vectorizer(5000, "english", max_df = 1.0, min_df = 0)
# using default values instead
review_vec = build_vectorizer(5000, "english")
# create tfidf matrix
perfume_by_term = review_vec.fit_transform(
    d['review'] for d in formatted_data).toarray()
index_to_vocab = {i: v for i, v in enumerate(review_vec.get_feature_names())}
all_ids = list(perfume_json["name"].keys())
# print(index_to_vocab)
all_perf_data = perfume_json_to_all_notes(perfume_json, all_ids)
name_to_index = perfume_name_to_index(all_perf_data)
index_to_id = perfume_index_to_id(all_perf_data)

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
        if perf_name.lower() == name.lower():
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


@app.route("/rocchio")
def new_results():

    pass


# this is main route combining query and gender preference
@app.route("/similar")
def similar_search():
    result = []

    name = request.args.get("name")
    gender_pref = request.args.get("gender_pref")
    min_rating = request.args.get("min_rating")
    rel_list = request.args.getlist("rel_list", type=str)
    irrel_list = request.args.getlist("irrel_list", type=str)

    print("name: " + name)
    print("gender_pref:" + gender_pref)
    print("min_rating: "+min_rating)
    print("rel_list: " + str(rel_list))
    print("irrel_list: " + str(irrel_list))

    exists = False
    for _, perf_name in perfume_json["name"].items():
        if perf_name.lower() == name.lower():
            exists = True
    if not exists:
        return json.dumps(result)
    # check if name is in perfume json, if not return "sorry pick another name"
    # Rest of algorithm goes here
    # 1. gender filter
    gendered_ids = gender_filter(perfume_json, all_ids, gender_pref)
    # 2. rating threshold filter
    rated_ids = rating_threshold_filter(perfume_json, gendered_ids, min_rating)

    # 3. jaccard sim filter
    num_perfumes = len(rated_ids)
    perf_data = perfume_json_to_all_notes(perfume_json, rated_ids)
    # jaccard = build_perf_sims_jac(num_perfumes, perf_data)
    # perfume_ind_to_id = perfume_index_to_id(perf_data)
    # jacc_ranked = get_ranked_perfumes(name, jaccard, perfume_ind_to_id, perf_data)

    # 4. rocchio filter
    rel_tuple = (name, rel_list)
    irrel_tuple = (name, irrel_list)

    # jaccard on 5 perufmes
    jaccard = build_perf_sims_jac(num_perfumes, perf_data)
    jacc_ranked = get_ranked_perfumes(name, jaccard, index_to_id, perf_data)

    # if len(rel_list) == 0 and len(irrel_list) == 0:
    # result = results(jacc_ranked, perfume_json)
    # result = initial_search(jacc_ranked, perfume_json)
    # else:
    cos_ranked = with_rocchio(
        rel_tuple, irrel_tuple, perfume_by_term, name_to_index, index_to_id, rocchio)
    combined_ranked = scores(jacc_ranked, cos_ranked, index_to_id)
    result = results(combined_ranked, perfume_json, name)

    return json.dumps(result)


"""
DO NOT DELETE: CODE FROM https://github.com/Y1chenYao/thank-u-next-cornell-prof-recommender/blob/master/backend/app.py
note: functions for edit distance in dropdowns
"""


def perfume_name_suggest(input_perf):
    perf_scores = {}
    perf_name_list = get_perfume_names(perfume_json)
    # perf_list is json file
    for perf in perf_name_list:
        score = fuzz.partial_ratio(input_perf.lower(), perf.lower())
        perf_scores[perf] = score
    sorted_perfs = sorted(perf_scores.items(),
                          key=lambda x: x[1], reverse=True)[:5]
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
def rating_threshold_filter(perfume_data, perfume_ids, threshold):
    res = []
    for id in perfume_ids:
        if (perfume_data["rating"][str(id)]) != "NA" and float(perfume_data["rating"][str(id)]) > float(threshold):
            res.append(id)
    return res

# THIS IS THE FILTERED PERFUME DATA TO USE

# get query perfume


def check_query(input_query, perf_json):
    # query = input_query.lower()
    perfume_names = get_perfume_names(perf_json)
    if input_query in perfume_names:
        return input_query
    return "Sorry, no results found. Check your spelling or try a different perfume."


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
    perf_idx = perfume_name_to_index(all_perf_data)[perfume]
    # Get list of similarity scores for movie
    score_lst = matrix[perf_idx]
    perf_score_lst = [(perf_index_to_id[i], s)
                      for i, s in enumerate(score_lst)]
    # Do not account for movie itself in ranking
    perf_score_lst = perf_score_lst[:perf_idx] + perf_score_lst[perf_idx+1:]
    # Sort rankings by score
    # perf_score_lst = sorted(perf_score_lst, key=lambda x: -x[1])
    return perf_score_lst

# get the necessary information


def initial_search(top_5, perf_json):
    """
        Take in list of top 5 perfumes ids and get the corresponding info
        input_dict: list of dictionaries for each perfume - perf_dict

        Returns: 
    """

    top_5 = sorted(top_5, key=lambda x: -x[1])
    final = []
    for i in range(5):
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


def results(top_5, perf_json, query_perf_name):
    """
        Take in list of top 5 perfumes ids and get the corresponding info
        input_dict: list of dictionaries for each perfume - perf_dict

        Returns: 
    """
    final = []
    for i in range(5):
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
        keyword_list = get_common_keywords(
            query_perf_name, perf_json["name"][top_5[i][0]])
        info["similarkeyword"] = keyword_list
        # keyword_str = ""
        # for word in keyword_list:
        #     if keyword_str == "":
        #         keyword_str = keyword_str + word
        #     else:
        #         keyword_str = keyword_str + ", " + word
        # info["similarkeyword"] = keyword_str
        final.append(info)
    return final

# functioons for roccchio
# def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
#     """Returns a TfidfVectorizer object with the above preprocessing properties.
#         The term document matrix of the perfume reviews input_doc_mat[i][j] is the tfidf
#         of the perfume i for the word j.

#     Note: This function may log a deprecation warning. This is normal, and you
#     can simply ignore it.

#     Parameters
#     ----------
#     max_features : int
#         Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer
#         constructer.
#     stop_words : str
#         Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer.
#     max_df : float
#         Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer.
#     min_df : float
#         Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer.
#     norm : str
#         Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer.

#     Returns
#     -------
#     TfidfVectorizer
#         A TfidfVectorizer object with the given parameters as its preprocessing properties.
#     """
#     v = TfidfVectorizer(max_features=max_features, stop_words=stop_words,
#                         max_df=max_df, min_df=min_df, norm=norm)
#     return v


def rocchio(perf, relevant, irrelevant, input_doc_matrix,
            perf_name_to_index, a=.3, b=.3, c=.8, clip=True):
    """Returns a vector representing the modified query vector. 

    Note: 
        If the `clip` parameter is set to True, the resulting vector should have 
        no negatve weights in it!

        Also, be sure to handle the cases where relevant and irrelevant are empty lists.

    Params: {query: String (the name of the movie being queried for),
             relevant: List (the names of relevant movies for query),
             irrelevant: List (the names of irrelevant movies for query),
             input_doc_matrix: Numpy Array,
             movie_name_to_index: Dict,
             a,b,c: floats (weighting of the original query, relevant queries,
                             and irrelevant queries, respectively),
             clip: Boolean (whether or not to clip all returned negative values to 0)}
    Returns: Numpy Array 
    """
    aq0 = a * input_doc_matrix[perf_name_to_index[perf]]

    rel_len = len(relevant)
    if rel_len == 0:
        rel_fraq = 0
    else:
        rel_fraq = 1/rel_len

    b_rel = np.zeros(len(aq0))
    for i in relevant:
        perfume = input_doc_matrix[perf_name_to_index[i]]
        b_rel = np.add(b_rel, perfume)

    b_rel = b_rel * b * rel_fraq

    nrel_len = len(irrelevant)
    if nrel_len == 0:
        nrel_fraq = 0
    else:
        nrel_fraq = 1/nrel_len

    c_nrel = np.zeros(len(aq0))
    for i in irrelevant:
        perfume = input_doc_matrix[perf_name_to_index[i]]
        c_nrel = np.add(c_nrel, perfume)

    c_nrel = c_nrel * c * nrel_fraq

    q1 = np.zeros(len(aq0))
    q1 = aq0 + b_rel - c_nrel
    return np.clip(q1, 0, None)


def with_rocchio(relevant_in, irrelevant_in, input_doc_matrix,
                 perf_name_to_index, perf_index_to_id, input_rocchio):
    """Returns a list in the following format:

        [(perf1, score1), (perf2, score2)..., (perf10,score10)],



    Parameters
    ----------
    relevant_in : (query: str, [relevant documents]: str list) list 
        tuple of the form:
        tuple[0] = name of perfume being queried (str), 
        tuple[1] = list of names of the relevant perfumes to the movie being queried (str list).
    irrelevant_in : (query: str, [irrelevant documents]: str list) list 
        The same format as relevant_in except tuple[1] contains list of irrelevant movies instead.
    input_doc_matrix : np.ndarray
        The term document matrix of the perfume reviews. input_doc_mat[i][j] is the tfidf
        of the perfume i for the word j.
    perf_name_to_index : dict
         A dictionary linking the perf name (Key: str) to the perf index (Value: int). 
         Ex: {'perf_0': 0, 'perfume_1': 1, .......}
    perf_index_to_name : dict
         A dictionary linking the perf index (Key: int) to the perf name (Value: str). 
         Ex: {0:'perf_0', 1:'perf_1', .......}
    input_rocchio: function
        A function implementing the rocchio algorithm.

    Returns
    -------
    dict
        Returns the top ten highest ranked perfumes and scores for each query in the format described above.

    """

    update = input_rocchio(
        relevant_in[0], relevant_in[1], irrelevant_in[1], input_doc_matrix, perf_name_to_index)

    sim = []
    for doc in input_doc_matrix:
        dots = np.dot(doc, update)
        q_norm = np.linalg.norm(doc)
        d_norm = np.linalg.norm(update)
        # fix div by zero error
        norm_prod = q_norm*d_norm if q_norm*d_norm != 0 else .0001
        sim.append(dots/(norm_prod))

    # indexes = np.argsort(sim)[::-1]
    # indexes = indexes[indexes != perf_name_to_index[relevant_in[0]]]
    sim = sim[:perf_name_to_index[relevant_in[0]]] + \
        sim[perf_name_to_index[relevant_in[0]]+1:]

    perfumes = []
    for ind in range(len(perf_name_to_index)-1):
        perfumes.append((perf_index_to_id[ind], sim[ind]))

    return perfumes


def scores(jaccard_in, cosine_in, perf_index_to_id):
    """
    Return top 5 of sorted rankings (most to least similar) of perfumes as
    a list of two-element tuples, where the first element is the
    perfume name and the second element is the combined similarity score 

    jaccard: list of jaccard similarity score (perfume id, score) tuples
        [(perf1, score1), (perf2, score2)..., (perf10,score10)]

    cosine: list of jaccard similarity score (perfume id, score) tuples
        [(perf1, score1), (perf2, score2)..., (perf10,score10)]
    """

    print("Jaccard_in: ")
    # print(jaccard_in)

    print("Cosine_in: ")
    # print(cosine_in)

    jac = []
    for tup in jaccard_in:
        jac.append(float(tup[1]))

    cos = []
    for tup in cosine_in:
        cos.append(float(tup[1]))

    jac = np.asarray(jac, dtype='float64')
    cos = np.asarray(cos, dtype='float64')

    # print("Jac: ")
    # print(jac)
    # print("Cos: ")
    # print(cos)

    scores = np.multiply(jac, cos)
    indexes = np.argsort(scores)[::-1]

    perfumes = []
    for ind in range(len(jaccard_in)):
        perfumes.append(
            (perf_index_to_id[indexes[ind]], scores[indexes[ind]]))

    return perfumes

    # IGNORE THIS

# def build_inverted_index(filtered_perf):
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
#     for i in range(len(filtered_perf)):
#         id = database[i]
#         notes = database['notes']
#         notes_set = set(notes)
#         for note in notes_set:
#             if note not in res:
#                 res[note] = []
#             res[note].append(i)
#     return res

    # funcctions for cosine sim

    # def build_query_word_counts(input):
    #     """ Builds an query_word_wounts from the messages.
    #     Arguments
    #     =========

    #     input: 1 string of a description

    #     Returns
    #     =======

    #     query_word_counts: dictionary
    #         For each term, the index contains
    #         1 item that represents the count of
    #         the term in the query  ==> count_of_term_in_doc
    #         query_word_counts[term] = count_of_term_in_doc

    #     """
    #    tokens = tokenizer.tokenize(input.lower())
    #     query_word_counts = {}

    #     for word in tokens:
    #         if word in query_word_counts:
    #             query_word_counts[word] += 1
    #         else:
    #             query_word_counts[word] = 1

    # def build_inverted_index(reviews):
    #     """ Builds an inverted index from the reviews of the desired perfume.

    #     Arguments
    #     =========

    #     reviews: list of dictionaries
    #         Each message in this list already has a 'toks'
    #         field that contains the tokenized message.
 #         [{1: { "toks: ["I", "love", "u"]}}, {2 : {"toks:["I", "hate", "u"]}, ...]
    #     Returns
    #     =======

    #     inverted_index: dict
    #         For each term, the index contains
    #         a sorted list of tuples (doc_id, count_of_term_in_doc)
    #         such that tuples with smaller doc_ids appear first:
    #         inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    #     Example
    #     =======

    #     >> test_idx = build_inverted_index([
    #     ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    #     ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])

    #     >> test_idx['be']
    #     [(0, 2), (1, 2)]

    #     >> test_idx['not']
    #     [(0, 1)]

    #     """
    #     # YOUR CODE HERE
    #     inverted_idx = {}

    #     for doc in range(len(reviews)-1):
    #         tokens = reviews[doc]["toks"]
    #         for word in tokens:
    #             if word in inverted_idx:
    #                 if doc in inverted_idx[word]:
    #                     inverted_idx[word][doc] += 1
    #                 else:
    #                     inverted_idx[word][doc] = 1
    #             else:
    #                 inverted_idx[word] = {}
    #                 inverted_idx[word][doc] = 1

    #     sort_inverted_idx = {}
    #     for word in inverted_idx:
    #         sort_list = list(inverted_idx[word].items())
    #         sort_list.sort(key=lambda i:i[0])
    #         sort_inverted_idx[word] = sort_list

    #     return sort_inverted_idx

    # def compute_idf(inv_idx, n_reviews, min_df=10, max_df_ratio=0.95):
    #     """ Compute term IDF values from the inverted index.
    #     Words that are too frequent or too infrequent get pruned.
    #     Hint: Make sure to use log base 2.
    #     Arguments
    #     =========

    #     inv_idx: an inverted index as above
    #     n_docs: int,
    #         The number of documents.
    #     min_df: int,
    #         Minimum number of documents a term must occur in.
    #         Less frequent words get ignored.
    #         Documents that appear min_df number of times should be included.
    #     max_df_ratio: float,
    #         Maximum ratio of documents a term can occur in.
    #         More frequent words get ignored.
    #     Returns
    #     =======

    #     idf: dict
    #         For each term, the dict contains the idf value.

    #     """

    #     # YOUR CODE HERE
    #     idf = {}

    #     for term in inv_idx:
    #         df = len(inv_idx[term])
    #         if df >= min_df and df/n_reviews <= max_df_ratio:
    #             ratio = n_reviews/(1+df)
    #             idf[term] = np.log2(ratio)

    #     return idf

    # def compute_doc_norms(index, idf, n_reviews):
    #     """ Precompute the euclidean norm of each document.
    #     Arguments
    #     =========
    #     index: the inverted index as above
    #     idf: dict,
    #         Precomputed idf values for the terms.
    #     n_docs: int,
    #         The total number of documents.
    #     Returns
    #     =======
    #     norms: np.array, size: n_docs
    #         norms[i] = the norm of document i.
    #     """

    #     # YOUR CODE HERE
    #     sums = np.zeros(n_reviews)
    #     for word in index:
    #         for (doc, tf) in index[word]:
    #             if word in idf:
    #                 sums[doc] += (tf*idf[word])**2
    #     norms = np.sqrt(sums)
    #     return norms

    # def accumulate_dot_scores(query_word_counts, index, idf):
    #     """ Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.
    #     Arguments
    #     =========

    #     query_word_counts: dict,
    #         A dictionary containing all words that appear in the query;
    #         Each word is mapped to a count of how many times it appears in the query.
    #         In other words, query_word_counts[w] = the term frequency of w in the query.
    #         You may safely assume all words in the dict have been already lowercased.

    #     index: the inverted index as above,

    #     idf: dict,
    #         Precomputed idf values for the terms.

    #     Returns
    #     =======

    #     doc_scores: dict
    #         Dictionary mapping from doc ID to the final accumulated score for that doc
    #     """

    #     doc_scores = {}

    #     for word in query_word_counts:
    #         if word in index:
    #             for (doc, tf) in index[word]:
    #                 if doc in doc_scores:
    #                     doc_scores[doc] += (query_word_counts[word] * idf[word]) * (tf * idf[word])
    #                 else:
    #                     doc_scores[doc] = (query_word_counts[word] * idf[word]) * (tf * idf[word])

    #     return doc_scores

    # def index_search(query, index, idf, doc_norms, score_func=accumulate_dot_scores, tokenizer=treebank_tokenizer):
    #     """ Search the collection of documents for the given query

    #     Arguments
    #     =========

    #     query: string,
    #         The query we are looking for.

    #     index: an inverted index as above

    #     idf: idf values precomputed as above

    #     doc_norms: document norms as computed above

    #     score_func: function,
    #         A function that computes the numerator term of cosine similarity (the dot product) for all documents.
    #         Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
    #         (See Q7)

    #     tokenizer: a TreebankWordTokenizer

    #     Returns
    #     =======

    #     results, list of tuples (score, doc_id)
    #         Sorted list of results such that the first element has
    #         the highest score, and `doc_id` points to the document
    #         with the highest score.

    #     Note:

    #     """

    #     tokens = tokenizer.tokenize(query.lower())
    #     query_word_counts = {}

    #     for word in tokens:
    #         if word in query_word_counts:
    #             query_word_counts[word] += 1
    #         else:
    #             query_word_counts[word] = 1

    #     scores = score_func(query_word_counts, index, idf)

    #     q_norm = 0
    #     for word in query_word_counts:
    #         if word in idf:
    #             q_norm += (query_word_counts[word]*idf[word])**2
    #     q_norm = np.sqrt(q_norm)

    #     for i in scores:
    #         scores[i] = scores[i]/(doc_norms[i]*q_norm)

    #     final_scores = []
    #     for tup in list(scores.items()):
    #         final_scores.append(tuple(reversed(tup)))

    #     final_scores.sort(key=lambda item: item[0], reverse=True)

    #     return final_scores[:5]

    # rev_results(top_5, perf_json)
    # """
    #     Take in list of top 5 perfumes ids and get the corresponding info
    #     top_5: list of dictionaries for each perfume - perf_dict

    #     Returns:
    # """
    # final = []
    # for i in range(len(top_5)):
    #     info = {}
    #     info["img"] = perf_json["image"][top_5[i][0]]
    #     info["gender"] = perf_json["for_gender"][top_5[i][0]]
    #     info["name"] = perf_json["name"][top_5[i][0]]
    #     info["brand"] = perf_json["company"][top_5[i][0]]
    #     info["rating"] = perf_json["rating"][top_5[i][0]]
    #     info["gender"] = perf_json["for_gender"][top_5[i][0]]
    #     info["topnote"] = perf_json["top notes"][top_5[i][0]]
    #     info["middlenote"] = perf_json["middle notes"][top_5[i][0]]
    #     info["bottomnote"] = perf_json["base notes"][top_5[i][0]]
    #     info["desc"] = perf_json["description"][top_5[i][0]]
    #     info
    #     final.append(info)
    # return final

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
