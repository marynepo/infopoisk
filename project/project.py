import os
import gensim
from gensim.summarization.bm25 import BM25
from gensim.models import KeyedVectors, Word2Vec
import json
from scipy import sparse
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.spatial.distance as distance
import streamlit as st
#from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import pickle
from time import time
import nltk
import joblib

nltk.download('stopwords')
sw = stopwords.words('russian')
m = Mystem()

def preprocess_text(text):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    words = [w for w in words if w not in sw]
    text = m.lemmatize(' '.join(words))
    return ''.join(text)[:-1]


def ind_query(qr, vct_name, vct):
    pr_qr = preprocess_text(qr)
    if vct_name in ['tfidf', 'cv', 'bm25']:
        return vct.transform([pr_qr]).toarray()
    elif vct_name == 'fasttext':
        return np.mean(vct[qr.split()], axis=0)
    else:
        t = vct[1](qr, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = vct[0](**{k: v.to(vct[0].device) for k, v in t.items()})
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings)[0].numpy()
        return embeddings


def calc_distance(qr, ind_mtx):
    dist = 1 - distance.cdist(ind_mtx, qr, 'cosine')
    return dist


def get_top(qr, vct_name, vct, answers, mtx):
    ind_q = ind_query(qr, vct_name, vct)
    if vct_name == 'bert':
        ind_q = ind_q.reshape(ind_q.shape[0], 1).T
    sort_ind = calc_distance(ind_q, mtx).argsort(axis=0)[-1::-1]
    return np.array(answers)[sort_ind].tolist()[:5]


@st.cache
def load_data():
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    vct_dt = {}
    for vct_name in ['bm25', 'tfidf', 'cv', 'fasttext', 'bert']:
        with open(vct_name + '_data', 'rb') as f:
            vct_dt[vct_name] = joblib.load(f)
    return data['ans'], vct_dt


def make_interface():
    st.title('Поисковик по корпусу ответов mail.ru')
    answers, vct_dt = load_data()
    with st.form(key='my_form'):
        qr = st.text_input(label='Введите запрос')
        vct_name = st.selectbox('Выберите векторайзер:', ['BM25', 'Tfidf', 'CV', 'FastText', 'Bert'])
        submit_button = st.form_submit_button(label='ОК')
    if submit_button:
        strt = time()
        res = get_top(qr, vct_name.lower(), vct_dt[vct_name.lower()]['vec'], answers, vct_dt[vct_name.lower()]['mtx'])
        fnsh = time()
        st.write(res)
        st.write('Время выполнения - ', fnsh - strt)


make_interface()
