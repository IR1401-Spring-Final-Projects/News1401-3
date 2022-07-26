import fasttext
import os
import numpy as np
import hazm
import re
from string import punctuation
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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


class Transformer:

    def __init__(self, preprocessor=None):
        self.model = SentenceTransformer('HooshvareLab/bert-fa-zwnj-base')
        self.preprocessor = preprocessor
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))
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

    def save_embeddings(self, path='Transformer_model.pickle'):
        with open(path, 'wb') as file:
            pickle.dump(self.embeddings, file)

    def load_embeddings(self, path='Transformer_model.pickle'):
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
        relevant_docs_mean = np.mean([self.model.encode(list([title])) for title in dataset.iloc[prelim_I.flatten().tolist()[:k]]['title']], axis=0)
        irrelevant_docs_mean = np.mean([self.model.encode(list([title])) for title in dataset.iloc[prelim_I.flatten().tolist()[-k:]]['title']], axis=0)
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
