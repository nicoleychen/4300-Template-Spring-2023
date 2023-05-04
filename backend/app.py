import numpy as np
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

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

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template('base.html', title="sample html")

# TODO: add def
# app.run(debug=True)

# JJ: function that opens json file
def load_perfume_data():
    f = open('perfumes_combined_0_499.json')
    data = json.load(f)
    print("JSON succesfully loaded!")
    f.close()
    return data


perfume_json = load_perfume_data()

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
        if count == 10:
            break
        word = index_to_vocab[word_id]
        if not (word.isnumeric()) and diff[word_id] != 0:
            keywords.append(word)
            count += 1
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


formatted_data = format_json(perfume_json)
stopwords_list = requests.get(
    "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines())
review_vec = build_vectorizer(5000, stopwords)
# create tfidf matrix
perfume_by_term = review_vec.fit_transform(
    d['review'] for d in formatted_data).toarray()
index_to_vocab = {i: v for i, v in enumerate(review_vec.get_feature_names())}
all_ids = list(perfume_json["name"].keys())
all_perf_data = perfume_json_to_all_notes(perfume_json, all_ids)
name_to_index = perfume_name_to_index(all_perf_data)
index_to_id = perfume_index_to_id(all_perf_data)


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
    rated_ids = rating_threshold_filter(
        perfume_json, gendered_ids, min_rating)

    query_id = str(index_to_id[name_to_index[name]])

    if not (query_id in rated_ids):
        rated_ids.append(query_id)
        rated_ids = sorted(rated_ids,  key=lambda x: int(x))

    # 3. jaccard sim filter
    num_perfumes = len(rated_ids)
    perf_data = perfume_json_to_all_notes(perfume_json, rated_ids)
    filtered_index_to_id = perfume_index_to_id(perf_data)
    # jaccard = build_perf_sims_jac(num_perfumes, perf_data)
    # perfume_ind_to_id = perfume_index_to_id(perf_data)
    # jacc_ranked = get_ranked_perfumes(name, jaccard, perfume_ind_to_id, perf_data)

    # 4. rocchio filter
    rel_tuple = (name, rel_list)
    irrel_tuple = (name, irrel_list)

    # jaccard on 5 perufmes
    jaccard = build_perf_sims_jac(num_perfumes, perf_data)
    jacc_ranked = get_ranked_perfumes(
        name, jaccard, filtered_index_to_id, perf_data)

    # if len(rel_list) == 0 and len(irrel_list) == 0:
    # result = results(jacc_ranked, perfume_json)
    # result = initial_search(jacc_ranked, perfume_json)
    # else:
    cos_ranked = with_rocchio(
        rel_tuple, irrel_tuple, perfume_by_term, name_to_index, index_to_id, rocchio)
    combined_ranked = scores(jacc_ranked, cos_ranked,
                             index_to_id, 1000)
    result = results(combined_ranked, perfume_json, name, 5)

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
def get_ranked_perfumes(perfume, matrix, filtered_perf_index_to_id, filtered_perf):
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
    # perf_idx = perfume_name_to_index(all_perf_data)[perfume]
    perf_idx = perfume_name_to_index(filtered_perf)[perfume]

    # Get list of similarity scores for movie
    score_lst = matrix[perf_idx]
    perf_score_lst = [(filtered_perf_index_to_id[i], s)
                      for i, s in enumerate(score_lst)]
    # Do not account for movie itself in ranking
    perf_score_lst = perf_score_lst[:perf_idx] + perf_score_lst[perf_idx+1:]
    # Sort rankings by score
    return perf_score_lst


def results(ranked_ids, perf_json, query_perf_name, top_k):
    """
        Take in list of ranked perfumes ids and get the corresponding info

        Returns: information for top_k perfumes
    """
    final = []
    for i in range(top_k):
        info = {}
        info["img"] = perf_json["image"][ranked_ids[i][0]]
        info["gender"] = perf_json["for_gender"][ranked_ids[i][0]]
        info["name"] = perf_json["name"][ranked_ids[i][0]]
        info["brand"] = perf_json["company"][ranked_ids[i][0]]
        info["rating"] = perf_json["rating"][ranked_ids[i][0]]
        info["gender"] = perf_json["for_gender"][ranked_ids[i][0]]
        info["topnote"] = perf_json["top notes"][ranked_ids[i][0]]
        info["middlenote"] = perf_json["middle notes"][ranked_ids[i][0]]
        info["bottomnote"] = perf_json["base notes"][ranked_ids[i][0]]
        # description = perf_json["description"][ranked_ids[i][0]]
        # no_languages = description[:description.find('Read')]
        info["desc"] = perf_json["description"][ranked_ids[i][0]]

        keyword_list = get_common_keywords(
            query_perf_name, perf_json["name"][ranked_ids[i][0]])
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
    sim = sim[:perf_name_to_index[relevant_in[0]]] + \
        sim[perf_name_to_index[relevant_in[0]]+1:]

    perfumes = []
    for ind in range(len(perf_name_to_index)-1):
        perfumes.append((perf_index_to_id[ind], sim[ind]))

    return perfumes


def scores(jaccard_in, cosine_in, perf_index_to_id, n):
    """
    Return top 5 of sorted rankings (most to least similar) of perfumes as
    a list of two-element tuples, where the first element is the
    perfume name and the second element is the combined similarity score 

    jaccard: list of jaccard similarity score (perfume id, score) tuples
        [(perf1, score1), (perf2, score2)..., (perf10,score10)]

    cosine: list of jaccard similarity score (perfume id, score) tuples
        [(perf1, score1), (perf2, score2)..., (perf10,score10)]
    """

    # score matrix where score_mat[0][i] contains jaccard sim score of perfume with id i
    # and score_mat[1][j] contains cosine sim score of perfume with id j
    # use 1000 for now to prevent index out of bound error
    score_mat = [[0 for i in range(n)] for j in range(2)]

    # jac = []
    for tup in jaccard_in:
        score_mat[0][int(tup[0])] = tup[1]

    # cos = []
    for tup in cosine_in:
        score_mat[1][int(tup[0])] = tup[1]


    scores = np.multiply(score_mat[0],  score_mat[1])

    perfumes = []
    for id, score in enumerate(scores):
        perfumes.append((str(id), score))

    perfumes_sorted = sorted(perfumes, key=lambda x: -x[1])

    return perfumes_sorted

   