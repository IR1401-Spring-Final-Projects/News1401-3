from pickle import FALSE
from shutil import ExecError
from unicodedata import category
from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseRedirect
from django.db import connection
from django.urls import reverse
from django.core.exceptions import FieldError
from django.db.models import Field
from django.apps import apps
import pickle
import zipfile
import os
from matplotlib.style import context
import pandas as pd
import csv
from mir.model_classes import *
import tqdm
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import codecs




TABLES = [x.__name__ for x in apps.get_models() if not x.__name__.startswith('Auth') and not x.__name__.startswith('Django')][:-6]
REQUEST_QUERY_KEY = 'query'
REQUEST_K_KEY = 'k'
BUTTON_SEARCH_BOOLEAN_KEY = 'boolean'
BUTTON_SEARCH_TF_IDF_KEY = 'tfidf'
BUTTON_SEARCH_FASTTEXT_KEY = 'fasttext'
BUTTON_SEARCH_TRANSFORMER_KEY = 'transformer_search'
BUTTON_SEARCH_ELASTIC_KEY = 'elastic'
BUTTON_CLASSIFY_LOGISTIC_REGRESSION_KEY = 'logistic_regression'
BUTTON_CLASSIFY_TRANSFORMER_KEY = 'transformer_classification'
BUTTON_CLUSTER_KEY = 'cluster'
CHECK_EXPANSION_KEY = 'queryexpansion'
BUTTON_LINK = 'linkanalysis'

INITIALIZED = False

BOOLEAN_MODEL = None
TF_IDF_MODEL = None
FASTTEXT_MODEL = None
TRANSFORMER_MODEL = None
ELASTIC_MODEL = None
LOGISTIC_REGRESSION_MODEL = None
TF_IDF_LR_MODEL = None
TRANSFORMER_CLASSIFICATION_MODEL = None
TRANSFORMER_CLASSIFICATION_TOKENIZER = None
CLUSTER_MODEL = None
LINK_ANALYSIS = None
DATASET = None
PREPROCESSED_TEXT = None
MINI_1K_DATASET = None
MINI_1K_PREPROCESSED_TEXT = None
MINI_4K_DATASET = None
MINI_4K_PREPROCESSED_TEXT = None
MINI_10K_DATASET = None
MINI_10K_PREPROCESSED_TEXT = None
PREPROCESSOR = Preprocessor(stopwords_path='mir/models/stopwords.txt')

def read_dataset_from_file():
    dataset = []
    with zipfile.ZipFile('mir/models/dataset.zip', 'r') as zip_file:
        zip_file.extractall()
    with open('dataset.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            data = dict(zip(header[1:], row[1:]))
            dataset.append(data)
    os.remove('dataset.csv')
    return pd.DataFrame(dataset)

def data_to_text(data):
    return ' '.join([data['title'], data['intro'], data['body']]).lower()

def get_mini_dataset(len_each_category=400):
    global CATEGORIES, DATASET, PREPROCESSOR

    mini_dataset = []
    for category in CATEGORIES.keys():
        dataset_by_category = DATASET.loc[DATASET['category'] == category]
        length = min(len_each_category, dataset_by_category.shape[0])
        mini_dataset.append(dataset_by_category.sample(length, random_state=1))

    mini_dataset = pd.concat(mini_dataset).reset_index(drop=True)
    texts = [data_to_text(data) for _, data in mini_dataset.iterrows()]
    mini_preprocessed_texts = [PREPROCESSOR.preprocess(text) for text in texts]
    return mini_dataset, mini_preprocessed_texts




def init_models():
    global DATASET, PREPROCESSED_TEXT, MINI_4K_DATASET, MINI_4K_PREPROCESSED_TEXT, MINI_10K_DATASET, MINI_10K_PREPROCESSED_TEXT,\
            MINI_1K_DATASET, MINI_1K_PREPROCESSED_TEXT, FASTTEXT_MODEL, LOGISTIC_REGRESSION_MODEL, TF_IDF_LR_MODEL, ELASTIC_MODEL, TF_IDF_MODEL, CLUSTER_MODEL, TRANSFORMER_CLASSIFICATION_MODEL, \
            TRANSFORMER_CLASSIFICATION_TOKENIZER, TRANSFORMER_MODEL, LINK_ANALYSIS, BOOLEAN_MODEL, ELASTIC_MODEL
    DATASET = read_dataset_from_file()
    with open('mir/models/Preprocessed_texts.pickle', "rb") as file:
        PREPROCESSED_TEXT = pickle.load(file)
    MINI_1K_DATASET, MINI_1K_PREPROCESSED_TEXT = get_mini_dataset(100)
    with open('mir/models/4k_dataset.pickle', "rb") as file:
            MINI_4K_DATASET, MINI_4K_PREPROCESSED_TEXT = pickle.load(file)
    with open('mir/models/10k_dataset.pickle', "rb") as file:
            MINI_10K_DATASET, MINI_10K_PREPROCESSED_TEXT = pickle.load(file)
    # BOOLEAN_MODEL = BooleanIR().prepare(MINI_1K_PREPROCESSED_TEXT, mode='load', save=False)
    # TF_IDF_MODEL = TF_IDF().prepare(MINI_1K_PREPROCESSED_TEXT, mode='load', save=False)
    # FASTTEXT_MODEL = FastText()
    # FASTTEXT_MODEL.prepare(MINI_4K_PREPROCESSED_TEXT, mode='load', save=False)
    TRANSFORMER_MODEL = Transformer()
    TRANSFORMER_MODEL.prepare(DATASET, mode='load', save=False)
    #ELASTIC_MODEL = News_Elasticsearch(1, 2, MINI_10K_DATASET, MINI_10K_PREPROCESSED_TEXT)
    # with open('mir/models/Logistic_Regression.pickle', "rb") as file:
    #         LOGISTIC_REGRESSION_MODEL = pickle.load(file)
    # TF_IDF_LR_MODEL = TF_IDF_LR()
    # TF_IDF_LR_MODEL.load_TF_IDF_model()
    # TRANSFORMER_CLASSIFICATION_MODEL = AutoModelForSequenceClassification.from_pretrained("mir/models/Transformer_Classification")
    # TRANSFORMER_CLASSIFICATION_TOKENIZER = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
    # with open('mir/models/KMeans_model.pickle', "rb") as file:
    #         CLUSTER_MODEL = pickle.load(file)
    # with open('mir/models/Link_analysis.pickle', "rb") as file:
    #         LINK_ANALYSIS = pickle.load(file)
    
    

def home(request):
    global INITIALIZED
    if not INITIALIZED:
        init_models()
        INITIALIZED = True
    context = {
        'tables': TABLES,
        'header': [],
        'header_title': [],
        'data': [],
        'category': [],
        'plot': [],
        'data_h': [],
        'data_a': [],
    }
    if request.method == 'POST':
        if BUTTON_SEARCH_BOOLEAN_KEY in request.POST:
            search_boolean(request, context)
        elif BUTTON_SEARCH_TF_IDF_KEY in request.POST:
            search_tf_idf(request, context)
        elif BUTTON_SEARCH_FASTTEXT_KEY in request.POST:
            search_fasttext(request, context)
        elif BUTTON_SEARCH_TRANSFORMER_KEY in request.POST:
            search_transformer(request, context)
        elif BUTTON_SEARCH_ELASTIC_KEY in request.POST:
            search_elastic(request, context)
        elif BUTTON_CLASSIFY_LOGISTIC_REGRESSION_KEY in request.POST or BUTTON_CLASSIFY_TRANSFORMER_KEY in request.POST:
            classify(request, context)
            return render(request, 'mir/classification.html', context)
        elif BUTTON_CLUSTER_KEY in request.POST:
            cluster(request, context)
            return render(request, 'mir/cluster.html', context)
        elif BUTTON_LINK in request.POST:
            link_analysis(request, context)
            return render(request, 'mir/link_analysis.html', context)
    return render(request, 'mir/home.html', context)




def cluster(request, context):
    global PREPROCESSOR, FASTTEXT_MODEL, TSNE_MODEL, TSNE_DATASET, MINI_4K_DATASET
    if request.method == 'POST':
        query = request.POST.get(REQUEST_QUERY_KEY, None)
        preprocessed_query = ' '.join(PREPROCESSOR.preprocess(query))
        query_embed = np.mean(list(map(lambda word: FASTTEXT_MODEL.model.get_word_vector(word), preprocessed_query)), axis=0)
        cluster_label = CLUSTER_MODEL.predict([query_embed.astype('float64')])[0]
        k = int(request.POST.get(REQUEST_K_KEY, None))
        result = []
        for idx, row in MINI_4K_DATASET.iterrows():
            if len(result) == k:
                break
            if CLUSTER_MODEL.labels_[idx] == cluster_label:
                result.append(row)
        result = pd.DataFrame(result)
        result['category'] = result['category'].apply(lambda x: CATEGORIES[x])
        context['category'] = cluster_label
        context['header_title'] = ['عنوان', 'مقدمه', 'متن خبر', 'موضوع']
        context['header'] = ['title', 'intro', 'body', 'category']
        context['data'] = result.to_dict('records')
        with codecs.open('./mir/templates/mir/cluster_dataset.html', 'rb', 'utf-8') as file:
            context['plot'] = file.read()

def classify(request, context):
    global PREPROCESSOR, LOGISTIC_REGRESSION_MODEL, TF_IDF_LR_MODEL, TRANSFORMER_CLASSIFICATION_MODEL, TRANSFORMER_CLASSIFICATION_TOKENIZER
    if request.method == 'POST':
        query = request.POST.get(REQUEST_QUERY_KEY, None)
        preprocessed_query = ' '.join(PREPROCESSOR.preprocess(query))
        if BUTTON_CLASSIFY_LOGISTIC_REGRESSION_KEY in request.POST:
            query_embed = TF_IDF_LR_MODEL.vectorizer.transform([preprocessed_query])
            predicted_class_code = LOGISTIC_REGRESSION_MODEL.predict(query_embed)[0]
            context['category'] = CATEGORIES[CLASSES_CATEGORIES[predicted_class_code]]
            probabilities = list(LOGISTIC_REGRESSION_MODEL.predict_proba(query_embed)[0])
        elif BUTTON_CLASSIFY_TRANSFORMER_KEY in request.POST:
            TRANSFORMER_CLASSIFICATION_MODEL.eval()
            device = torch.cuda.current_device() if torch.cuda.is_available() else -1
            pipe = TextClassificationPipeline(model=TRANSFORMER_CLASSIFICATION_MODEL, tokenizer=TRANSFORMER_CLASSIFICATION_TOKENIZER, return_all_scores=True, device=device)
            predictions = pipe(preprocessed_query)[0]
            probabilities = [prediction['score'] for prediction in predictions]
            context['category'] = CATEGORIES[CLASSES_CATEGORIES[np.argmax(probabilities)]]
        context['header_title'] = list(CATEGORIES.values())
        context['header_title'].insert(0, 'موضوع')
        context['header'] = np.arange(len(context['header_title']))
        data = []
        probabilities.insert(0, 'احتمال')
        temp = {context['header'][j]:probabilities[j] for j in range(len(context['header']))}
        for k, v in temp.items():
            if v != 'احتمال':
                temp[k] = round(v, 3)
        data.append(temp)
        context['data'] = data
    
def search_boolean(request, context):
    global CATEGORIES
    query = request.POST.get(REQUEST_QUERY_KEY, None)
    k = int(request.POST.get(REQUEST_K_KEY, None))
    expansion = request.POST.get(CHECK_EXPANSION_KEY, False)
    context['header_title'] = ['عنوان', 'مقدمه', 'متن خبر', 'موضوع']
    context['header'] = ['title', 'intro', 'body', 'category']
    if expansion:
        result = BOOLEAN_MODEL.predict_with_expansion([query], MINI_1K_DATASET, k=k)
    else:
        result = BOOLEAN_MODEL.predict([query], MINI_1K_DATASET, k=k)
    if type(result) == str:
        context['data'] = [{}]
    else:
        result['category'] = result['category'].apply(lambda x: CATEGORIES[x])
        context['data'] = result.to_dict('records')

def search_tf_idf(request, context):
    global CATEGORIES
    query = request.POST.get(REQUEST_QUERY_KEY, None)
    k = int(request.POST.get(REQUEST_K_KEY, None))
    expansion = request.POST.get(CHECK_EXPANSION_KEY, False)
    context['header_title'] = ['عنوان', 'مقدمه', 'متن خبر', 'موضوع']
    context['header'] = ['title', 'intro', 'body', 'category']
    if expansion:
        result = TF_IDF_MODEL.predict_with_expansion(query, MINI_1K_DATASET, k=k)
    else:
        result = TF_IDF_MODEL.predict(query, MINI_1K_DATASET, k=k)
    result['category'] = result['category'].apply(lambda x: CATEGORIES[x])
    context['data'] = result.to_dict('records')

def search_fasttext(request, context):
    global CATEGORIES
    query = request.POST.get(REQUEST_QUERY_KEY, None)
    k = int(request.POST.get(REQUEST_K_KEY, None))
    expansion = request.POST.get(CHECK_EXPANSION_KEY, False)
    context['header_title'] = ['عنوان', 'مقدمه', 'متن خبر', 'موضوع']
    context['header'] = ['title', 'intro', 'body', 'category']
    if expansion:
        result = FASTTEXT_MODEL.predict_with_expansion(query, MINI_4K_DATASET, k=k)
    else:
        result = FASTTEXT_MODEL.predict(query, MINI_4K_DATASET, k=k)
    result['category'] = result['category'].apply(lambda x: CATEGORIES[x])
    context['data'] = result.to_dict('records')


def search_transformer(request, context):
    global CATEGORIES
    query = request.POST.get(REQUEST_QUERY_KEY, None)
    k = int(request.POST.get(REQUEST_K_KEY, None))
    expansion = request.POST.get(CHECK_EXPANSION_KEY, False)
    context['header_title'] = ['عنوان', 'مقدمه', 'متن خبر', 'موضوع']
    context['header'] = ['title', 'intro', 'body', 'category']
    if expansion:
        result = TRANSFORMER_MODEL.predict_with_expansion(query, DATASET, k=k)
    else:
        result = TRANSFORMER_MODEL.predict(query, DATASET, k=k)
    result['category'] = result['category'].apply(lambda x: CATEGORIES[x])
    context['data'] = result.to_dict('records')

def search_elastic(request, context):
    global CATEGORIES
    query = request.POST.get(REQUEST_QUERY_KEY, None)
    k = int(request.POST.get(REQUEST_K_KEY, None))
    context['header_title'] = ['عنوان', 'مقدمه', 'متن خبر', 'موضوع']
    context['header'] = ['title', 'intro', 'body', 'category']
    result = ELASTIC_MODEL.search(query, k=k)
    result['category'] = result['category'].apply(lambda x: CATEGORIES[x])
    context['data'] = result.to_dict('records')
    


def link_analysis(request, context):
    global LINK_ANALYSIS
    print(list(LINK_ANALYSIS.keys()))
    k = int(request.POST.get(REQUEST_K_KEY, None))
    vals = np.array(list(LINK_ANALYSIS['pr'].values()))
    idx = np.argsort(-vals)
    pr_df = MINI_10K_DATASET.iloc[idx[:k]]
    pr_df['category'] = pr_df['category'].apply(lambda x: CATEGORIES[x])
    context['data'] = pr_df.to_dict('records')
    vals_h = np.array(list(LINK_ANALYSIS['h'].values()))
    idx_h = np.argsort(-vals_h)
    vals_a = np.array(list(LINK_ANALYSIS['a'].values()))
    idx_a = np.argsort(-vals_a)
    h_df = MINI_10K_DATASET.iloc[idx_h[:k]]
    h_df['category'] = h_df['category'].apply(lambda x: CATEGORIES[x])
    context['data_h'] = h_df.to_dict('records')
    a_df = MINI_10K_DATASET.iloc[idx_a[:k]]
    a_df['category'] = a_df['category'].apply(lambda x: CATEGORIES[x])
    context['data_a'] = a_df.to_dict('records')
    context['header_title'] = ['عنوان', 'مقدمه', 'متن خبر', 'موضوع']
    context['header'] = ['title', 'intro', 'body', 'category']


def about(request):
    return render(request, 'mir/about.html')
