import os
import re
import hazm
import tqdm
import torch
import faiss
import pickle
import fasttext
import numpy as np
import pandas as pd
from string import punctuation
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    TextClassificationPipeline, AutoTokenizer

from elasticsearch import Elasticsearch

CATEGORIES = {
    'Politics': 'سیاسی',
    'World': 'جهانی',
    'Economy': 'اقتصادی',
    'Society': 'اجتماعی',
    'City': 'شهری',
    'Sport': 'ورزشی',
    'Science': 'علمی',
    'Culture': 'فرهنگی',
    'IT': 'فناوری اطلاعات',
    'LifeSkills': 'مهارت‌های زندگی',
}

CATEGORIES_CLASSES = {
    'Politics': 0,
    'World': 1,
    'Economy': 2,
    'Society': 3,
    'City': 4,
    'Sport': 5,
    'Science': 6,
    'Culture': 7,
    'IT': 8,
    'LifeSkills': 9,
}

CLASSES_CATEGORIES = {v: k for k, v in CATEGORIES_CLASSES.items()}


class Preprocessor:

    def __init__(self, stopwords_path):
        self.stopwords = []
        with open(stopwords_path, encoding='utf-8') as file:
            self.stopwords = file.read().split()

    def preprocess(self, text):
        text = self.normalize(text)
        text = self.remove_links(text)
        text = self.remove_punctuations(text)
        words = self.word_tokenize(text)
        words = self.remove_invalid_words(words)
        words = self.remove_stopwords(words)
        return words

    def normalize(self, text):
        return hazm.Normalizer().normalize(text)

    def remove_links(self, text):
        patterns = ['\S*http\S*', '\S*www\S*', '\S+\.ir\S*', '\S+\.com\S*', '\S+\.org\S*', '\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, ' ', text)
        return text

    def remove_punctuations(self, text):
        return re.sub(f'[{punctuation}؟،٪×÷»«]+', '', text)

    def word_tokenize(self, text):
        return hazm.word_tokenize(text)

    def remove_invalid_words(self, words):
        return [word for word in words if len(word) > 3 or re.match('^[\u0600-\u06FF]{2,3}$', word)]

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]


class TF_IDF_LR:

    def __init__(self, vocabulary=None):
        self.vectorizer = TfidfVectorizer(vocabulary=vocabulary)

    def fit_transform_vectorizer(self, dataset):
        vectors = self.vectorizer.fit_transform(list(map(lambda doc: ' '.join(doc), dataset)))
        dense_vectors = vectors.todense().tolist()
        return np.array(dense_vectors)

    def save_TF_IDF_model(self, path="mir/models/TF_IDF_LR.pickle"):
        with open(path, "wb") as file:
            pickle.dump(self.vectorizer, file)

    def load_TF_IDF_model(self, path="mir/models/TF_IDF_LR.pickle"):
        with open(path, "rb") as file:
            self.vectorizer = pickle.load(file)


class TF_IDF:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.words = None
        self.dense_vectors_df = None

    def fit_transform_vectorizer(self, dataset):
        self.vectors = self.vectorizer.fit_transform(list(map(lambda doc: ' '.join(doc), dataset)))
        self.words = self.vectorizer.get_feature_names_out()
        dense_vectors = self.vectors.todense().tolist()
        self.dense_vectors_df = pd.DataFrame(dense_vectors, columns=self.words)

    def predict(self, query, dataset, k):
        query = ' '.join(Preprocessor(stopwords_path='mir/models/stopwords.txt').preprocess(query))
        query_transform = self.vectorizer.transform([query]).todense().tolist()[0]
        dense_vectors = self.dense_vectors_df.values.tolist()
        df_cosine_sim = list(map(lambda doc: self.cosine_sim(query_transform, doc), dense_vectors))
        self.dense_vectors_df['query_sim'] = df_cosine_sim
        indices = self.dense_vectors_df.nlargest(k, 'query_sim').index
        self.dense_vectors_df = self.dense_vectors_df.drop(columns=['query_sim'])
        return dataset.iloc[indices]

    def expand_query(self, query, k=5, lambda_0=1, lambda_1=1):
        query = ' '.join(Preprocessor(stopwords_path='mir/models/stopwords.txt').preprocess(query))
        query_transform = self.vectorizer.transform([query]).todense().tolist()[0]
        dense_vectors = np.array(self.dense_vectors_df.values.tolist())
        dataset_sim = np.array(list(map(lambda doc: self.cosine_sim(query_transform, doc), dense_vectors)))
        idx = np.argsort(-dataset_sim)
        relevant_docs_mean = np.mean(dense_vectors[list(idx[:k]), :], axis=0)
        irrelevant_docs_mean = np.mean(dense_vectors[list(idx[-k:]), :], axis=0)
        final_embed = query_transform + lambda_0 * relevant_docs_mean - lambda_1 * irrelevant_docs_mean
        return final_embed

    def predict_with_expansion(self, query, dataset, k):
        expanded_query_embed = self.expand_query(query)
        dense_vectors = self.dense_vectors_df.values.tolist()
        df_cosine_sim = list(map(lambda doc: self.cosine_sim(expanded_query_embed, doc), dense_vectors))
        self.dense_vectors_df['query_sim'] = df_cosine_sim
        indices = self.dense_vectors_df.nlargest(k, 'query_sim').index
        self.dense_vectors_df = self.dense_vectors_df.drop(columns=['query_sim'])
        return dataset.iloc[indices]

    def cosine_sim(self, query, doc):
        return np.dot(query, doc) / (np.linalg.norm(query) * np.linalg.norm(doc))

    def save_TF_IDF_model(self, path="mir/models/TF_IDF_model.pickle"):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load_TF_IDF_model(self, path="mir/models/TF_IDF_model.pickle"):
        with open(path, "rb") as file:
            return pickle.load(file)

    def prepare(self, dataset, mode, save=False):
        model = None
        if mode == 'train':
            self.fit_transform_vectorizer(dataset)
            model = self
        if mode == 'load':
            model = self.load_TF_IDF_model()
        if save:
            self.save_TF_IDF_model()
        return model


class Transformer:

    def __init__(self, preprocessor=None):
        self.model = SentenceTransformer('HooshvareLab/bert-fa-zwnj-base')
        self.preprocessor = preprocessor
        # if torch.cuda.is_available():
        #     self.model = self.model.to(torch.device('cuda'))
        self.embeddings = None
        self.index = None

    def train_embeddings(self, train_dataset: list):
        if type(train_dataset[0]) == list:
            train_dataset = list(map(lambda doc: ' '.join(doc), train_dataset))
        self.embeddings = self.model.encode(train_dataset, show_progress_bar=True)
        self.embeddings = np.array(
            [embedding for embedding in self.embeddings]).astype('float32')

    def make_index(self, dataset: list):
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(self.embeddings, np.arange(len(dataset)).astype('int64'))

    def save_embeddings(self, path='mir/models/Transformer_model.pickle'):
        with open(path, 'wb') as file:
            pickle.dump(self.embeddings, file)

    def load_embeddings(self, path='mir/models/Transformer_model.pickle'):
        with open(path, 'rb') as file:
            self.embeddings = pickle.load(file)

    def predict(self, query: str, dataset: pd.DataFrame, k=10):
        if self.preprocessor:
            query = ' '.join(self.preprocessor.preprocess(query))
        vector = self.model.encode(list([query]))
        D, I = self.index.search(np.array(vector).astype('float32'), k=k)
        return dataset.iloc[I.flatten().tolist()]

    def expand_query(self, query, dataset, k=5, lambda_0=1, lambda_1=1):
        if self.preprocessor:
            query = ' '.join(self.preprocessor.preprocess(query))
        query_embed = self.model.encode(list([query]))
        prelim_D, prelim_I = self.index.search(np.array(query_embed).astype('float32'), k=len(dataset))
        relevant_docs_mean = np.mean(
            [self.model.encode(list([title])) for title in dataset.iloc[prelim_I.flatten().tolist()[:k]]['title']],
            axis=0)
        irrelevant_docs_mean = np.mean(
            [self.model.encode(list([title])) for title in dataset.iloc[prelim_I.flatten().tolist()[-k:]]['title']],
            axis=0)
        final_embed = query_embed + lambda_0 * relevant_docs_mean - lambda_1 * irrelevant_docs_mean
        return final_embed

    def predict_with_expansion(self, query, dataset, k):
        expanded_query_embed = self.expand_query(query, dataset)
        D, I = self.index.search(np.array(expanded_query_embed).astype('float32'), k=k)
        return dataset.iloc[I.flatten().tolist()]

    def prepare(self, dataset, mode, save=False):
        if mode == 'train':
            self.train_embeddings(dataset)
        if mode == 'load':
            self.load_embeddings()
        if save:
            self.save_embeddings()
        self.make_index(dataset)


class FastText:

    def __init__(self, preprocessor=None, method='skipgram'):
        self.method = method
        self.mean_embed = []
        self.model = None
        self.preprocessor = preprocessor

    def train(self, texts):
        with open('FastText_train.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(list(map(lambda doc: ' '.join(doc), texts))))
        self.model = fasttext.train_unsupervised('FastText_train.txt', self.method, minn=2, maxn=5, wordNgrams=10)
        os.remove('FastText_train.txt')
        self.mean_embed = list(map(lambda doc:
                                   np.mean(list(map(lambda word:
                                                    self.model.get_word_vector(word), doc)), axis=0), texts))
        self.mean_embed = np.array(self.mean_embed)

    def predict(self, query, dataset, k):
        if self.preprocessor:
            query = self.preprocessor.preprocess(query)
        if type(query) == str:
            query = query.split()
        query_embed = np.mean(list(map(lambda word: self.model.get_word_vector(word), query)), axis=0)
        dataset_sim = np.array(list(map(lambda doc: self.cosine_sim(query_embed, doc), self.mean_embed)))
        idx = np.argsort(-dataset_sim)
        return dataset.iloc[list(idx[:k])]

    def expand_query(self, query, k=5, lambda_0=1, lambda_1=1):
        if self.preprocessor:
            query = self.preprocessor.preprocess(query)
        if type(query) == str:
            query = query.split()
        query_embed = np.mean(list(map(lambda word: self.model.get_word_vector(word), query)), axis=0)
        dataset_sim = np.array(list(map(lambda doc: self.cosine_sim(query_embed, doc), self.mean_embed)))
        idx = np.argsort(-dataset_sim)
        relevant_docs_mean = np.mean(self.mean_embed[list(idx[:k]), :], axis=0)
        irrelevant_docs_mean = np.mean(self.mean_embed[list(idx[-k:]), :], axis=0)
        final_embed = query_embed + lambda_0 * relevant_docs_mean - lambda_1 * irrelevant_docs_mean
        return final_embed

    def predict_with_expansion(self, query, dataset, k):
        expanded_query_embed = self.expand_query(query)
        dataset_sim = np.array(list(map(lambda doc: self.cosine_sim(expanded_query_embed, doc), self.mean_embed)))
        idx = np.argsort(-dataset_sim)
        return dataset.iloc[list(idx[:k])]

    def cosine_sim(self, query, doc):
        return np.dot(query, doc) / (np.linalg.norm(query) * np.linalg.norm(doc))

    def save_FastText_model(self, path='mir/models/FastText_model.bin'):
        self.model.save_model(path)
        np.save('mir/models/FastText_mean_embed.npy', self.mean_embed)

    def load_FastText_model(self, path="mir/models/FastText_model.bin"):
        self.model = fasttext.load_model(path)
        self.mean_embed = np.load('mir/models/FastText_mean_embed.npy')

    def prepare(self, dataset, mode, save=False):
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_FastText_model()
        if save:
            self.save_FastText_model()


class News_Elasticsearch:

    def __init__(self, dataset, preprocessed_texts, username, password):
        os.environ['ES_ENDPOINT'] = f"http://{username}:{password}@localhost:9200"
        self.es = Elasticsearch(os.environ['ES_ENDPOINT'])
        if self.es.indices.exists(index='news'):
            self.es.indices.delete(index='news')
        else:
            self.es.indices.create(index='news')
        self.dataset = dataset
        self.index_dataset(preprocessed_texts)

    def index_dataset(self, preprocessed_texts):
        for i in range(len(preprocessed_texts)):
            text = ' '.join(preprocessed_texts[i])
            self.es.index(index='news', id=i, document={'text': text})

    def search(self, preprocessed_query, k):
        query = {
            'match': {
                'text': preprocessed_query
            }
        }
        hits = self.es.search(index='news', query=query, size=k)['hits']['hits']
        indices = [hit['_id'] for hit in hits]
        return self.dataset.iloc[indices]


class BooleanIR:

    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)
        self.vectors = None
        self.words = None
        self.dense_vectors_df = None

    def fit_transform_vectorizer(self, dataset):
        self.vectors = self.vectorizer.fit_transform(list(map(lambda doc: ' '.join(doc), dataset)))
        self.words = self.vectorizer.get_feature_names_out()
        dense_vectors = self.vectors.todense().tolist()
        self.dense_vectors_df = pd.DataFrame(dense_vectors, columns=self.words)

    def preprocess_query(self, query):
        preprocessor = Preprocessor(stopwords_path='mir/models/stopwords.txt')
        preprocessed_query = []
        for subquery in query:
            if subquery[0] == '-':
                preprocessed_query.append('-' + (' '.join(preprocessor.preprocess(subquery[1:]))))
            else:
                preprocessed_query.append(' '.join(preprocessor.preprocess(subquery)))
        return preprocessed_query

    def predict(self, query, dataset, k, expansion=False):
        if not expansion:
            query = self.preprocess_query(query)
        query_result = []
        query_result_BUT = []
        for subquery in query:
            is_subquery_but = False
            if subquery[0] == '-':
                is_subquery_but = True
                subquery = subquery[1:]
            if expansion:
                query_transform = query[0]
            else:
                query_transform = np.array(self.vectorizer.transform([subquery]).todense().tolist()[0])
            query_indices = np.argwhere(query_transform >= 0.5)
            if len(query_indices) > 0:
                query_indices = query_indices[0]
            else:
                return 'چیزی برای نمایش نیست! :('
            for index, doc in self.dense_vectors_df.iterrows():
                if all(np.take(doc, query_indices)):
                    if is_subquery_but:
                        query_result_BUT.append(index)
                    else:
                        query_result.append(index)
        query_result = list(set(query_result) - set(query_result_BUT))
        k = min(k, len(query_result))
        return dataset.iloc[query_result[:k]]

    def expand_query(self, query, dataset, lambda_0=0.5, lambda_1=0.5):
        query = ' '.join(Preprocessor(stopwords_path='mir/models/stopwords.txt').preprocess(query))
        query_transform = self.vectorizer.transform([query]).todense().tolist()[0]
        dense_vectors = np.array(self.dense_vectors_df.values.tolist())
        dataset_sim = np.array(list(map(lambda doc: self.cosine_sim(query_transform, doc), dense_vectors)))
        idx = np.argsort(-dataset_sim)
        relevant_docs_mean = np.mean(dense_vectors[list(idx[:2]), :], axis=0)
        irrelevant_docs_mean = np.mean(dense_vectors[list(idx[-2:]), :], axis=0)
        final_embed = query_transform + lambda_0 * relevant_docs_mean - lambda_1 * irrelevant_docs_mean
        return final_embed

    def predict_with_expansion(self, query, dataset, k):
        expanded_query = self.expand_query(query[0], dataset)
        return self.predict([expanded_query], dataset, k, expansion=True)

    def cosine_sim(self, query, doc):
        return np.dot(query, doc) / (np.linalg.norm(query) * np.linalg.norm(doc))

    def save_boolean_model(self, path='mir/models/BooleanIR_model.pickle'):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load_boolean_model(self, path='mir/models/BooleanIR_model.pickle'):
        with open(path, "rb") as file:
            return pickle.load(file)

    def prepare(self, dataset, mode, save=False):
        model = None
        if mode == 'train':
            self.fit_transform_vectorizer(dataset)
            model = self
        if mode == 'load':
            model = self.load_boolean_model()
        if save:
            self.save_boolean_model()
        return model
