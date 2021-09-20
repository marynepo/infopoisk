sw = stopwords.words('russian')
m = Mystem()


def make_corpus(path):
    docs = []
    docnames = []
    for root, dirs, files in os.walk(path):
        for name in files:
            filepath = os.path.join(root, name)
            with open(filepath, 'r') as f:
                docs.append(f.read())
                docnames.append(name)
    return docs, docnames

def preprocess_text(text):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    words = [w for w in words if w not in sw]
    text = m.lemmatize(' '.join(words))
    return ''.join(text)[:-1]
