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


def main():
    a, q = make_corpus('data.jsonl')
    with open('data.pickle', 'wb') as f:
        pickle.dump({'ans': a, 'quest': q}, f)

    mt, vc = index_mtx(a)
    for mx, v in zip(mt.items(), vc.items()):
        with open(mx[0] + '_data.pickle', 'wb') as f:
            pickle.dump({'mtx': mx[1], 'vct': v[1]}, f)

    for v_n, v in tqdm(vc.items()):
        i_q = ind_query(q, v_n, v)
        with open(v_n + '_qs.pickle', 'wb') as f:
            pickle.dump(i_q, f)

if __name__=='__main__':
    main()