from pickle import FALSE
from shutil import ExecError
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
import pandas as pd
import csv
from model_classes import *
import tqdm

TABLES = [x.__name__ for x in apps.get_models() if not x.__name__.startswith('Auth') and not x.__name__.startswith('Django')][:-6]
REQUEST_QUERY_KEY = 'query'
REQUEST_K_KEY = 'k'
BUTTON_SEARCH_KEY = 'search'
BUTTON_CLASSIFY_KEY = 'classify'
BUTTON_CLUSTER_KEY = 'cluster'
BUTTON_CANCEL_KEY = 'cancel'

INITIALIZED = False

BOOLEAN_MODEL = None
TF_IDF_MODEL = None
FASTTEXT_MODEL = None
TRANSFORMER_MODEL = None
LOGISTIC_REGRESSION_MODEL = None
TRANSFORMER_CLASSIFICATION_MODEL = None
CLUSTER_MODEL = None
TSNE_MODEL = None
TSNE_DATASET = None
DATASET = None
PREPROCESSED_TEXT = None
MINI_4K_DATASET = None
MINI_4K_PREPROCESSED_TEXT = None
MINI_10K_DATASET = None
MINI_10K_PREPROCESSED_TEXT = None
PREPROCESSOR = Preprocessor(stopwords_path='models/stopwords.txt')

def read_dataset_from_file():
    dataset = []
    with zipfile.ZipFile('dataset.zip', 'r') as zip_file:
        zip_file.extractall()
    with open('dataset.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            data = dict(zip(header[1:], row[1:]))
            dataset.append(data)
    os.remove('dataset.csv')
    return dataset

def data_to_text(data):
    return ' '.join([data['title'], data['intro'], data['body']]).lower()

def get_mini_dataset(len_each_category=400):
    global CATEGORIES, dataset

    mini_dataset = []
    for category in CATEGORIES.keys():
        dataset_by_category = dataset.loc[dataset['category'] == category]
        length = min(len_each_category, dataset_by_category.shape[0])
        mini_dataset.append(dataset_by_category.sample(length, random_state=1))

    mini_dataset = pd.concat(mini_dataset).reset_index(drop=True)
    texts = [data_to_text(data) for _, data in mini_dataset.iterrows()]
    mini_preprocessed_texts = [PREPROCESSOR.preprocess(text) for text in tqdm(texts)]
    return mini_dataset, mini_preprocessed_texts


mini_dataset, mini_preprocessed_texts = get_mini_dataset()


def init_models():
    DATASET = read_dataset_from_file
    with open('models/Preprocessed_texts.pickle', "rb") as file:
        PREPROCESSED_TEXT = pickle.load(file)
    MINI_4K_DATASET, MINI_4K_PREPROCESSED_TEXT = get_mini_dataset()
    MINI_10K_DATASET, MINI_10K_PREPROCESSED_TEXT = get_mini_dataset(1000)
    with open('models/BooleanIR_model.pickle', "rb") as file:
            BOOLEAN_MODEL = pickle.load(file)
    with open('models/TF_IDF_model.pickle', "rb") as file:
            TF_IDF_MODEL = pickle.load(file)
    with open('models/Transformer_model.pickle', "rb") as file:
            TRANSFORMER_MODEL = pickle.load(file)
    with open('models/Logistic_Regression.pickle', "rb") as file:
            LOGISTIC_REGRESSION_MODEL = pickle.load(file)
    with open('models/Transformer_classification.pickle', "rb") as file:
            TRANSFORMER_CLASSIFICATION_MODEL = pickle.load(file)
    with open('models/KMeans.pickle', "rb") as file:
            CLUSTER_MODEL = pickle.load(file)
    with open('models/TSNE_model.pickle', "rb") as file:
            TSNE_MODEL = pickle.load(file)
    with open('models/TSNE_dataset.pickle', "rb") as file:
            TSNE_DATASET = pickle.load(file)
    FASTTEXT_MODEL = FastText()
    FASTTEXT_MODEL.prepare(MINI_4K_PREPROCESSED_TEXT, mode='load', save=False)
    
    

def home(request):
    if not INITIALIZED:
        init_models()
        INITIALIZED = True
    context = {
        'tables': TABLES,
        'header': [],
        'data': [],
    }
    if request.method == 'POST':
        if BUTTON_SEARCH_KEY in request.POST:
            search(request, context)
        elif BUTTON_CLUSTER_KEY in request.POST:
            return redirect('mir-cluster')
        elif BUTTON_CLASSIFY_KEY in request.POST:
            return redirect('mir-cluster')
    return render(request, 'mir/home.html', context)




def cluster(request):
    if request.method == 'POST':
        if BUTTON_CANCEL_KEY in request.POST:
            return redirect('mir-home')
        else:
            query = request.POST.get(REQUEST_QUERY_KEY, None)
    # plot dataset and query
    # plt.to_html()
    return render(request, 'mir/cluster.html')

def classify(request):
    if request.method == 'POST':
        if BUTTON_CANCEL_KEY in request.POST:
            return redirect('mir-home')
        else:
            query = request.POST.get(REQUEST_QUERY_KEY, None)

    return render(request, 'mir/cluster.html')


def search(request, context):
    query = request.POST.get(REQUEST_QUERY_KEY, None)
    k = request.POST.get(REQUEST_K_KEY, None)
    
    context['header'] = ['Title', 'Intro', 'Text', 'Category']
    temp = []
    for i in range(int(k)):
        temp.append({'Title': 1, 'Intro' : 2, 'Text' : 3, 'Category' : 4})
    context['data'] = temp


def about(request):
    return render(request, 'mir/about.html')
