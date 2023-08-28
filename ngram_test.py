from kiwipiepy import Kiwi
from collections import defaultdict
from pprint import pprint
kiwi = Kiwi(num_workers=4, model_type='sbg')

def tokenizer_kiwi(sent, pre_spacing=False:  # [(형태소1, 품사1), (형태소2, 품사2), ...] 형태로 결과를 리턴
    if pre_spacing:
        sent = kiwi.space(sent, reset_whitespace=True)
    result = list(map(lambda x:x[0] + "/" + x[1], kiwi.tokenize(sent)))

    return result

# source code from https://lovit.github.io/nlp/2018/10/23/ngram/
# from konlpy.tag import Komoran
# komoran = Komoran()
# words = komoran.pos(test3, join=True)
def to_ngrams(words, n):
    ngrams = []
    for b in range(0, len(words) - n + 1):
        ngrams.append(tuple(words[b:b+n]))
    return ngrams

def get_ngram_counter(docs, min_count=10, n_range=(1,4)):  # docs = [doc1, doc2 ...]
    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in docs:
        print(doc)
        words = tokenizer_kiwi(doc) #  komoran.pos(doc, join=True)  # 코모란으로. # ['버들붕어/NNP', '개체군/NNP', '내/NP' ... ]
        for n in range(n_begin, n_end + 1):
            for ngram in to_ngrams(words, n):
                ngram_counter[ngram] += 1

    ngram_counter = {
        ngram: count for ngram, count in ngram_counter.items()
        if count >= min_count
    }

    return ngram_counter

def get_ngram_score(ngram_counter, delta=30):
    ngrams_ = {}
    for ngram, count in ngram_counter.items():
        if len(ngram) == 1:
            continue
        print(ngram)
        first = ngram_counter[ngram[:-1]]
        second = ngram_counter[ngram[1:]]
        score = (count - delta) / (first * second)
        if score > 0:
            ngrams_[ngram] = (count, score)
    return ngrams_

ngram_counter = get_ngram_counter(outparallel_splited, min_count=10, n_range=(1,4))

ngram_scores = get_ngram_score(ngram_counter, delta=10)

# 높은 단어 점수의 bi/tri.. gram 보기
trigram_scores = {
    ngram:score for ngram, score in ngram_scores.items()
    if len(ngram) == 3
}

pprint(sorted(trigram_scores.items(), key=lambda x:-x[1][1])[:30])  # trigram 뽑기

# 빈도 / POS template 로 filtering
def get_matched_ngram_counter(docs, templates, non_template, min_count=5, n_range=(2,5)):

    def to_ngrams(words, n):
        ngrams = []
        for b in range(0, len(words) - n + 1):
            ngrams.append(tuple(words[b:b+n]))
        return ngrams

    def find_matched_ngram(ngrams):
        matcheds = []
        for ngram in ngrams:
            for template in templates:
                if match(ngram, template):
                    if match(ngram, non_template) == False:
                        matcheds.append(ngram)
        return matcheds

    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in docs:
        words = tokenizer_kiwi(doc)
        for n in range(n_begin, n_end + 1):
            ngrams = to_ngrams(words, n)
            ngrams = find_matched_ngram(ngrams)
            for ngram in ngrams:
                ngram_counter[ngram] += 1

    ngram_counter = {
        ngram:count for ngram, count in ngram_counter.items()
        if count >= min_count
    }

    return ngram_counter


templates = [
    # ('/NN', '/J', '/NN'),
    # ('/NN', '/NN'),
    # ('/NN', '/NN', '/NN'),
    # ('/NN', '/NN', '/NN', '/NN'),
    # ('/NN', '/NN', '/NN', '/NN', '/NN'),
    ('',) * 2 + tuple(['/NN']),
    ('',) * 3 + tuple(['/NN']),
    ('',) * 4 + tuple(['/NN'])
]

non_template = [
    ('')
]

matched_ngrams = get_matched_ngram_counter(outparallel_splited,
                                           templates,
                                           non_template,
                                           min_count=10,
                                           n_range=(2,5))

len(matched_ngrams)
pprint(matched_ngrams)

to_ngrams(['나', '는'], 1)

templates = [
    ('/NN',), ('/NN', '/NN'),
]
templates
def match(ngram, template):
    if len(ngram) != len(template):
        return False
    for n, t in zip(ngram, template):
        if not (t in n):
            return False
    return True

ngram = ('최고/NNG')
template = templates[0]
templates[0]
match(ngram, template) # True