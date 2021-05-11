
# Vietnamese question answering system

Description of request:
- Enter question Q
- The system gives an answer A

---
## Using Pipe
```
Pipe = Pipeline([
    ('bow',CountVectorizer(analyzer=cleaner)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])
```
```
Pipe.predict(['Còn tiền không?'])[0]
' Còn chết liền '
```
---
## Using Word2Vec

Use baomoi.model.binWord2Vec Vietnamese

You can find model [here](https://thiaisotajppub.s3-ap-northeast-1.amazonaws.com/publicfiles/baomoi.model.bin)

Initialize a model with e.g
```
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
```
Find the top-N most similar words.

```
from gensim.models import KeyedVectors
model_path = './baomoi.model.bin'
w2v = KeyedVectors.load_word2vec_format(model_path, fvocab=None, binary=True, encoding='utf8')
print(w2v.similar_by_word('anh'))
```
---

