import pickle
with open('/Users/kintch/Library/CloudStorage/Dropbox/Mac/Downloads/for_embedding (2).pkl','rb') as f:
    unconnected_word_list_2023_ = pickle.load(f)
    unconnected_word_list_2022_ = pickle.load(f)
    unconnected_word_list_2021_ = pickle.load(f)

    connected_word_list_tbs_2023_over3_ = pickle.load(f)
    connected_word_list_tbs_2022_over3_ = pickle.load(f)
    connected_word_list_tbs_2021_over3_ = pickle.load(f)

    unconnected_word_list_2023_over3_ = pickle.load(f)
    unconnected_word_list_2022_over3_ = pickle.load(f)
    unconnected_word_list_2021_over3_ = pickle.load(f)

    freq_tbs_2023 = pickle.load(f)
    freq_tbs_2022 = pickle.load(f)
    freq_tbs_2021 = pickle.load(f)

    freq_tbs_2023_over3 = pickle.load(f)
    freq_tbs_2022_over3 = pickle.load(f)
    freq_tbs_2021_over3 = pickle.load(f)

# test unfamiliar words



import numpy as np
from soynlp.hangle import compose, decompose, character_is_korean
import re


# test -------------------------------------------------
# ft_mdl_path = '/Users/kintch/Downloads/cc.ko.300.bin'
# # model = FastText.load(model_fname)
#
# # use gensim
# from gensim.models import fasttext
# model = fasttext.load_facebook_model(ft_mdl_path, encoding = 'utf-8')
#
# import fasttext
# import fasttext.util
# ft = fasttext.load_model(ft_mdl_path)
# ft.get_nearest_neighbors('브레턴우즈')
# ------------------------------------------------------


from gensim.models import FastText
model_fname = '/Users/kintch/Downloads/jamoed_wiki_n_namu_based_model'
model = FastText.load(model_fname)

def jamo_sentence(sent):
    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        if len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else '-' for c in cjj)
        return cjj_

    sent_ = []
    for char in sent:
        if character_is_korean(char):
            sent_.append(transform(char))
        else:
            sent_.append(char)
    sent_ = doublespace_pattern.sub(' ', ''.join(sent_))
    return sent_

def jamo_to_word(jamo):
    jamo_list, idx = [], 0
    while idx < len(jamo):
        if not character_is_korean(jamo[idx]):
            jamo_list.append(jamo[idx])
            idx += 1
        else:
            jamo_list.append(jamo[idx:idx + 3])
            idx += 3
    word = ""
    for jamo_char in jamo_list:
        if len(jamo_char) == 1:
            word += jamo_char
        elif jamo_char[2] == "-":
            word += compose(jamo_char[0], jamo_char[1], " ")
        else:
            word += compose(jamo_char[0], jamo_char[1], jamo_char[2])
    return word

def transform(list):
    return [(jamo_to_word(w), r) for (w, r) in list]

doublespace_pattern = re.compile('\s+')

def similar_words(word, k):
    jamoed_word = jamo_sentence(word)
    result = transform(model.wv.most_similar(jamoed_word, topn=k))
    return result

# jamo-based word analogy test

# test_result = model.wv.evaluate_word_analogies(analogies='/Users/kintch/Documents/kor_analogy_result.txt')
# test_result[0]
# model.wv.evaluate_word_analogies(analogies='/Users/kintch/Documents/kor_analogy.txt')

# with open('/Users/kintch/Documents/kor_analogy.txt') as f, open('/Users/kintch/Documents/kor_analogy_result.txt', 'w') as g:
#     text = f.readlines()
#     for tline in text:
#         if ':' in tline:
#             g.writelines(tline+'\n')
#         else:
#             jamoed_word = jamo_sentence(tline)
#             g.writelines(jamoed_word+'\n')

# similar_words('1960년 트리핀', 10)
# similar_words('당시 대규모 대미 무역', 10)
# jamo_sentence('1960년트리핀')

def strred_vw_maker(word):
    # make wv string
    tmp=''
    for x in model.wv[jamo_sentence(word)].tolist():
        tmp += str(x)+'\t'
    tmp=tmp[:-1]
    return tmp

def cosine_similarity(word1, word2):
    cjj1 = jamo_sentence(word1)
    cjj2 = jamo_sentence(word2)
    cos_sim = model.wv.similarity(cjj1, cjj2)
    return cos_sim


# tb_soonong_words_ = list(tb_soonong_words)
# ebs_soonong_words_ = list(ebs_soonong_words)
# unfam_soonong_words_ = list(unfam_soonong_words)

connected_word_list_over3 = connected_word_list_tbs_2023_over3_\
                            + connected_word_list_tbs_2022_over3_\
                            + connected_word_list_tbs_2021_over3_
unconnected_word_list_over3 = unconnected_word_list_2023_over3_\
                              + unconnected_word_list_2022_over3_\
                              + unconnected_word_list_2021_over3_


with open('/Users/kintch/Downloads/wv_0814.tsv', 'w', encoding='utf-8') as f, open('/Users/kintch/Downloads/words_0814.tsv', 'w', encoding='utf-8') as v:
    v.writelines('word' + '\t' + 'source' + '\n')
    for tb in connected_word_list_over3:
        v.writelines(str(tb)+'\t'+'not unfamiliar'+'\n')
        f.writelines(strred_vw_maker(tb)+'\n')
    for tb in unconnected_word_list_over3:
        v.writelines(str(tb)+'\t'+'unfamiliar'+'\n')
        f.writelines(strred_vw_maker(tb)+'\n')


# 시각화(matplotlib)

# emb_vec = np.array([model.wv[jamo_sentence(tb)] for tb in unconnected_word_list_over3+connected_word_list_over3])
# label = np.array(['지나치게 생소한']*len(unconnected_word_list_over3)+['생소하지 않은']*len(connected_word_list_over3))
#
# # model.wv[jamo_sentence('지나치게')]
#
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# import seaborn as sns
# import pandas as pd
# iris = load_iris()
# x = iris.data
# y = iris.target
# x[:5]
# y[:5]
#
# tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=10, n_iter=30000, learning_rate=10)
# z = tsne.fit_transform(emb_vec)
#
# df = pd.DataFrame()
# df["y"] = label
# df["comp_1"] = z[:,0]
# df["comp_2"] = z[:,1]
# plt.rcParams['font.family'] ='AppleGothic'
# plt.rcParams['axes.unicode_minus'] =False
# sns.scatterplot(x="comp_1", y="comp_2", hue=df.y.tolist(), style=df.y.tolist(),
#                 palette=sns.color_palette("hls", 2),
#                 data=df, s=10).set(title="Iris data T-SNE projection")
#
# for line in range(0,df.shape[0]):
#      plt.text(df.comp_1[line]+0.2, df.comp_2[line], df.y[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
#
#
#
#
#
# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(x_mnist)
# df = pd.DataFrame()
# df["y"] = y_train
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]
#
# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 10),
#                 data=df).set(title="MNIST data T-SNE projection")


# jamo_sentence('대미 무역 흑자')
# similar_words('1960년 트리핀', 10)
# similar_words('당시 대규모 대미 무역', 10)
# cosine_similarity('당시 대규모 대미 무역', '대미무역 흑자')


# 조합생성
from itertools import product
from tqdm import tqdm
a=[1,2,3]
b=[4,5]
list(product(*[a, b]))

#교과서/ebs (2023년 기준)내 어휘
tbs_2023 = [word for word, freq in freq_tbs_2023.items()]
tbs_2022 = [word for word, freq in freq_tbs_2022.items()]
tbs_2021 = [word for word, freq in freq_tbs_2021.items()]

# 친숙하지 않은 어휘 - 교과서/ebs 어휘 간 similiarity 계산
from tqdm import tqdm
unfam_sim_2023=list()
for word1, word2 in tqdm(list(product(*[unconnected_word_list_2023_over3_,tbs_2023]))):
    # print(word1, word2)
    tmp = [word1, word2, cosine_similarity(word1, word2)]
    unfam_sim_2023.append(tmp)

unfam_sim_2022=list()
for word1, word2 in tqdm(list(product(*[unconnected_word_list_2022_over3_,tbs_2022]))):
    # print(word1, word2)
    tmp = [word1, word2, cosine_similarity(word1, word2)]
    unfam_sim_2022.append(tmp)

unfam_sim_2021=list()
for word1, word2 in tqdm(list(product(*[unconnected_word_list_2021_over3_,tbs_2021]))):
    # print(word1, word2)
    tmp = [word1, word2, cosine_similarity(word1, word2)]
    unfam_sim_2021.append(tmp)

# 친숙하지 않지 않은 어휘 - 교과서/ebs 어휘 간 similiarity 계산
fam_sim_2023=list()
for word1, word2 in tqdm(list(product(*[connected_word_list_tbs_2023_over3_,tbs_2023]))):
    # print(word1, word2)
    tmp = [word1, word2, cosine_similarity(word1, word2)]
    fam_sim_2023.append(tmp)

fam_sim_2022=list()
for word1, word2 in tqdm(list(product(*[connected_word_list_tbs_2022_over3_,tbs_2022]))):
    # print(word1, word2)
    tmp = [word1, word2, cosine_similarity(word1, word2)]
    fam_sim_2022.append(tmp)

fam_sim_2021=list()
for word1, word2 in tqdm(list(product(*[connected_word_list_tbs_2021_over3_,tbs_2021]))):
    # print(word1, word2)
    tmp = [word1, word2, cosine_similarity(word1, word2)]
    fam_sim_2021.append(tmp)


import pickle
with open('/Users/kintch/Library/CloudStorage/Dropbox/Mac/Downloads/cosim.pkl','wb') as f:
    pickle.dump(unfam_sim_2023, f)
    pickle.dump(unfam_sim_2022, f)
    pickle.dump(unfam_sim_2021, f)

import pandas as pd



def calculate_cossim_freq(unfam_word_list,
                          unfam_sim_list,
                          vocab_freq_dict):
    unfam_sim_df = pd.DataFrame(unfam_sim_list, columns=['word1', 'word2', 'cossim'])

    cossim_dot_freq=list()
    for word in tqdm(unfam_word_list):
        tmp = unfam_sim_df[unfam_sim_df.word1==word].sort_values(by=['cossim'], axis=0, ascending=False)
        tmp = tmp[:10]
        sim_list = tmp.cossim.tolist()
        freq_list = [np.log1p(vocab_freq_dict[x]) for x in tmp.word2.tolist()]

        #weight freq
        cossim_dot_freq.append(np.dot(sim_list, freq_list))

    return cossim_dot_freq




cossim_dot_freq_2023_unfam = calculate_cossim_freq(unconnected_word_list_2023_over3_, unfam_sim_2023, freq_tbs_2023)
cossim_dot_freq_2022_unfam = calculate_cossim_freq(unconnected_word_list_2022_over3_, unfam_sim_2022, freq_tbs_2022)
cossim_dot_freq_2021_unfam = calculate_cossim_freq(unconnected_word_list_2021_over3_, unfam_sim_2021, freq_tbs_2021)

cossim_dot_freq_2023_fam = calculate_cossim_freq(connected_word_list_tbs_2023_over3_, fam_sim_2023, freq_tbs_2023)
cossim_dot_freq_2022_fam = calculate_cossim_freq(connected_word_list_tbs_2022_over3_, fam_sim_2022, freq_tbs_2022)
cossim_dot_freq_2021_fam = calculate_cossim_freq(connected_word_list_tbs_2021_over3_, fam_sim_2021, freq_tbs_2021)


unfam_list_all = [unconnected_word_list_2023_over3_, unconnected_word_list_2022_over3_, unconnected_word_list_2021_over3_]
cossim_dot_freq_list = [cossim_dot_freq_2023_unfam, cossim_dot_freq_2022_unfam, cossim_dot_freq_2021_unfam]

fam_list_all = [connected_word_list_tbs_2023_over3_, connected_word_list_tbs_2022_over3_, connected_word_list_tbs_2021_over3_]
fam_cossim_dot_freq_list = [cossim_dot_freq_2023_fam, cossim_dot_freq_2022_fam, cossim_dot_freq_2021_fam]


list_all = [unconnected_word_list_2023_over3_+connected_word_list_tbs_2023_over3_,
            unconnected_word_list_2022_over3_+connected_word_list_tbs_2022_over3_,
            unconnected_word_list_2021_over3_+connected_word_list_tbs_2021_over3_]
cossim_dot_freq_list_all = [cossim_dot_freq_2023_unfam+cossim_dot_freq_2023_fam,
                            cossim_dot_freq_2022_unfam+cossim_dot_freq_2022_fam,
                            cossim_dot_freq_2021_unfam+cossim_dot_freq_2021_fam]

k=2023
for x, y in zip(list_all,cossim_dot_freq_list_all):
    test_df = pd.DataFrame({'word': x, 'realate_metric': y})
    test_df.to_excel('/Users/kintch/Library/CloudStorage/Dropbox/Mac/Downloads/all_cmetric_081216('+ str(k)+ ').xlsx')
    k+=-1


#test
# [(x,y) for x, y in zip([1,2,3], [4,5,6])]

k=2023
for x, y in zip(unfam_list_all,cossim_dot_freq_list):
    test_df = pd.DataFrame({'word': x, 'realate_metric': y})
    test_df.to_excel('/Users/kintch/Library/CloudStorage/Dropbox/Mac/Downloads/unfam_cmetric_081216('+ str(k)+ ').xlsx')
    k+=-1

k=2023
for x, y in zip(fam_list_all,fam_cossim_dot_freq_list):
    test_df = pd.DataFrame({'word': x, 'realate_metric': y})
    test_df.to_excel('/Users/kintch/Library/CloudStorage/Dropbox/Mac/Downloads/fam_cmetric_081216('+ str(k)+ ').xlsx')
    k+=-1

# 기술통계

pd.concat([
    pd.Series(cossim_dot_freq_2023_unfam+cossim_dot_freq_2022_unfam+cossim_dot_freq_2021_unfam, name='unfam').describe(),
    pd.Series(cossim_dot_freq_2023_fam+cossim_dot_freq_2022_fam+cossim_dot_freq_2021_fam, name='fam').describe(),
    pd.Series(cossim_dot_freq_2023_unfam, name='2023_unfam').describe(),
    pd.Series(cossim_dot_freq_2023_fam, name='2023_fam').describe(),
    pd.Series(cossim_dot_freq_2022_unfam, name='2022_unfam').describe(),
    pd.Series(cossim_dot_freq_2022_fam, name='2022_fam').describe(),
    pd.Series(cossim_dot_freq_2021_unfam, name='2021_unfam').describe(),
    pd.Series(cossim_dot_freq_2021_fam, name='2021_fam').describe(),
           ],
          axis=1).to_excel('/Users/kintch/Library/CloudStorage/Dropbox/Mac/Downloads/cmetric_081216(2022).xlsx')


pd.concat([
    pd.Series(cossim_dot_freq_2023_fam+cossim_dot_freq_2023_unfam, name='2023').describe(),
    pd.Series(cossim_dot_freq_2022_fam+cossim_dot_freq_2022_unfam, name='2022').describe(),
    pd.Series(cossim_dot_freq_2021_fam+cossim_dot_freq_2021_unfam, name='2021').describe(),
            ],
          axis=1).to_excel('/Users/kintch/Library/CloudStorage/Dropbox/Mac/Downloads/all_cmetric_081216.xlsx')