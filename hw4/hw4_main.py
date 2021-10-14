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
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
from tqdm import tqdm
from sklearn.preprocessing import normalize
import pickle


sw = stopwords.words('russian')
m = Mystem()


def make_corpus(path):
    with open(path, 'r') as f:
        corpus = list(f)[:10000]
    answers = []
    quests = []
    for doc in corpus:
        doc = json.loads(doc)
        ans = sorted(doc['answers'], key=lambda x: x['author_rating']['value'])
        if len(ans) > 0:
            answers.append(ans[-1]['text'])
            quests.append(doc['question'])
    return answers, quests


def preprocess_text(text):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    words = [w for w in words if w not in sw]
    text = m.lemmatize(' '.join(words))
    return ''.join(text)[:-1]


def index_mtx(docs):
    pr_docs = [preprocess_text(d) for d in tqdm(docs)]
    w2v = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    w2v_X = []
    for doc in tqdm(pr_docs):
        if len(doc) != 0:
            w2v_X.append(np.mean(w2v[doc.split()], axis=0))
        else:
            w2v_X.append(w2v[''])
    w2v_X = w2v_X
    tfidf = TfidfVectorizer()
    tfidf_X = tfidf.fit_transform(pr_docs).toarray()
    countv = CountVectorizer()
    counv_X = countv.fit_transform(pr_docs).toarray()
    bm25 = BM25([doc.split() for doc in pr_docs])
    bm25_X = []
    for doc in tqdm(pr_docs):
        bm25_X.append(bm25.get_scores(doc.split()))
    bm25_X = np.array(bm25_X)
    tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
    bert = AutoModel.from_pretrained('cointegrated/rubert-tiny').cuda()
    k = 0
    for doc in docs:
        t = tokenizer(doc, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = bert(**{k: v.to(bert.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)[0].cpu().numpy()
        if k == 0:
            bert_X = embeddings
            k = 1
        else:
            bert_X = np.vstack((bert_X,embeddings))
    vectorizers = {'tfidf': tfidf, 'cv': countv, 'bm25':bm25, 'w2v': w2v, 'bert': [bert, tokenizer]}
    mtxs = {'tfidf': tfidf_X, 'cv': counv_X, 'bm25': bm25_X, 'w2v': w2v_X, 'bert': bert_X}
    return mtxs, vectorizers


def ind_query(qrs, vct_name, vct):
    pr_qrs = [preprocess_text(qr) for qr in tqdm(qrs)]
    if vct_name == 'tfidf' or vct_name == 'cv':
        return vct.transform(pr_qrs).toarray()
    elif vct_name == 'bm25':
        res = []
        for qr in pr_qrs:
            res.append(vct.get_scores(qr.split()))
        return np.array(res)
    elif vct_name == 'w2v':
        res = []
        for qr in pr_qrs:
            if len(qr) != 0:
                res.append(np.mean(vct[qr.split()], axis=0))
            else:
                res.append(vct[''])
        return np.array(res)
    else:
        k = 0
        for qr in qrs:
            t = vct[1](qr, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = vct[0](**{k: v.to(vct[0].device) for k, v in t.items()})
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings)[0].cpu().numpy()
            if k == 0:
                bert_qr = embeddings
                k = 1
            else:
                bert_qr = np.vstack((bert_qr,embeddings))
        return bert_qr


def calc_distance(qr, ind_mtx):
    dist = np.dot(normalize(ind_mtx), normalize(qr).T)
    return dist


def get_top(qrs, vct_name, vct, answers, mtx, comp=False):
    if not comp:
        ind_q = ind_query(qrs, vct_name, vct)
    else:
        ind_q = qrs
    if len(qrs) == 1:
        if vct_name == 'bert':
            ind_q = ind_q.reshape(ind_q.shape[0], 1).T
        sort_ind = calc_distance(ind_q, mtx).argsort(axis=0)[-1:-6:-1]
        return np.array(answers)[sort_ind].tolist()
    sort_ind = calc_distance(ind_q, mtx).argsort(axis=0)
    return np.array(answers)[sort_ind][:, -1:-6:-1].tolist()


def compare_vct(docs, qrs):
    scores = {}
    for vname in tqdm(['tfidf', 'cv', 'bm25', 'w2v', 'bert']):
        with open(vname + '_qs.pickle', 'rb') as f:
            ind_q = pickle.load(f)
        with open(vname + '_data.pickle', 'rb') as f:
            dt = pickle.load(f)
        if 'mtxs' in dt:
            dt['mtx'] = dt['mtxs']
        tops = get_top(ind_q, vname, dt['vct'], docs, dt['mtx'], comp=True)
        scr = 0
        for doc, top in zip(docs, tops):
            if doc in top:
                scr += 1
        scores[v_info[0]] = scr / len(doc)
    return scores


def user_qr(answers):
    qr = input('Введите запрос или нажмите на Enter, если больше не хотите спрашивать: ')
    results = []
    while qr != '':
        vct_name = input('''Введите "cv", если хотите использовать каунтвекторайзер,  
                         "tfidf" - тфидф, "w2v" - ворд2век, "bm25" - бм25 и "bert" - берт ''')
        with open(vct_name + '_data.pickle') as f:
            dt = pickle.load(f)
        ind_m = dt['mtx']
        vctz = dt['vct']
        res = get_top([qr], vct_name, vctz, answers, ind_m)
        results.append(res)
        print('Топ 5:\n', '\n'.join([str(i) for i in res]))
        qr = input('Введите запрос или нажмите на Enter, если больше не хотите спрашивать: ')
    return results


def main():
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    answers = data['ans']
    questions = data['quest']
    goal = input(
        'Если вы хотите сравнить все векторайзеры введите 0, если же вы хотите задавать запрсоы - 1. Если вы больше ничего не хотите, нажмите Enter ')
    while goal != '':
        if goal == '0':
            comp_results = compare_vct(answers, questions)
            print(comp_results)
        else:
            qr_results = user_qr(answers)
        goal = input(
            'Если вы хотите сравнить все векторайзеры введите 0, если же вы хотите задавать запрсоы - 1. Если вы больше ничего не хотите, нажмите Enter ')

if __name__=='__main__':
    main()
