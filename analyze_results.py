import pickle
with open('/Users/kintch/PycharmProjects/ksat_nonfiction_challenge/variables.pkl','rb') as f:
    textbook_2015curri_vocab = pickle.load(f)
    ebs_2022_vocab = pickle.load(f)
    ebs_2021_vocab = pickle.load(f)
    ebs_2020_vocab = pickle.load(f)
    ebs_2019_vocab = pickle.load(f)
    ebs_2018_vocab = pickle.load(f)

    pos_2015curri_vocab = pickle.load(f)
    pos_ebs_2022_vocab = pickle.load(f)
    pos_ebs_2021_vocab = pickle.load(f)
    pos_ebs_2020_vocab = pickle.load(f)
    pos_ebs_2019_vocab = pickle.load(f)
    pos_ebs_2018_vocab = pickle.load(f)

    textbook_2015curri_vocab_over5 = pickle.load(f)
    ebs_2022_vocab_over5 = pickle.load(f)
    ebs_2021_vocab_over5 = pickle.load(f)
    ebs_2020_vocab_over5 = pickle.load(f)
    ebs_2019_vocab_over5 = pickle.load(f)
    ebs_2018_vocab_over5 = pickle.load(f)

    pos_2015curri_vocab_over5 = pickle.load(f)
    pos_ebs_2022_vocab_over5 = pickle.load(f)
    pos_ebs_2021_vocab_over5 = pickle.load(f)
    pos_ebs_2020_vocab_over5 = pickle.load(f)
    pos_ebs_2019_vocab_over5 = pickle.load(f)
    pos_ebs_2018_vocab_over5 = pickle.load(f)

    tb_soon_2022 = pickle.load(f)
    tb_soon_2021 = pickle.load(f)
    tb_soon_2020 = pickle.load(f)
    tb_soon_2019 = pickle.load(f)
    tb_soon_2018 = pickle.load(f)

    ebs_soon_2022 = pickle.load(f)
    ebs_soon_2021 = pickle.load(f)
    ebs_soon_2020 = pickle.load(f)
    ebs_soon_2019 = pickle.load(f)
    ebs_soon_2018 = pickle.load(f)

    tb_soon_2022_over5 = pickle.load(f)
    tb_soon_2021_over5 = pickle.load(f)
    tb_soon_2020_over5 = pickle.load(f)
    tb_soon_2019_over5 = pickle.load(f)
    tb_soon_2018_over5 = pickle.load(f)

    ebs_soon_2022_over5 = pickle.load(f)
    ebs_soon_2021_over5 = pickle.load(f)
    ebs_soon_2020_over5 = pickle.load(f)
    ebs_soon_2019_over5 = pickle.load(f)
    ebs_soon_2018_over5 = pickle.load(f)

    tokend_soon_2022 = pickle.load(f)
    tokend_soon_2021 = pickle.load(f)
    tokend_soon_2020 = pickle.load(f)
    tokend_soon_2019 = pickle.load(f)
    tokend_soon_2018 = pickle.load(f)
from kiwipiepy import Kiwi
def make_word_pos_list(word_pos_idx_list):
    result=[]
    for x in word_pos_idx_list:
        word_tuple = ()
        for y in x:
            word_tuple += (y[0:2],)
        # print(word_tuple)
        result.append(word_tuple)
    return result
# 형태소 분석 옵션 추가 정의 (add_pre_analyzed_word)
kiwi = Kiwi(num_workers=4, model_type='sbg')  #, typos='basic'
kiwi.add_pre_analyzed_word('실세계', [('실','NNG'), ('세계', 'NNG')], 100)
kiwi.add_pre_analyzed_word('격자판', [('격자', 'NNG'), ('판','NNG')], 100)
kiwi.add_pre_analyzed_word('신용도', [('신용', 'NNG'), ('도','NNG')], 100)
kiwi.add_user_word('내인성', 'NNP', 100)




tb_soon_2022_ = make_word_pos_list(tb_soon_2022)
tb_soon_2021_ = make_word_pos_list(tb_soon_2021)
tb_soon_2020_ = make_word_pos_list(tb_soon_2020)
tb_soon_2019_ = make_word_pos_list(tb_soon_2019)
tb_soon_2018_ = make_word_pos_list(tb_soon_2018)

ebs_soon_2022_ = make_word_pos_list(ebs_soon_2022)
ebs_soon_2021_ = make_word_pos_list(ebs_soon_2021)
ebs_soon_2020_ = make_word_pos_list(ebs_soon_2020)
ebs_soon_2019_ = make_word_pos_list(ebs_soon_2019)
ebs_soon_2018_ = make_word_pos_list(ebs_soon_2018)

# 빈도 5 이상

tb_soon_2022_over5_ = make_word_pos_list(tb_soon_2022_over5)
tb_soon_2021_over5_ = make_word_pos_list(tb_soon_2021_over5)
tb_soon_2020_over5_ = make_word_pos_list(tb_soon_2020_over5)
tb_soon_2019_over5_ = make_word_pos_list(tb_soon_2019_over5)
tb_soon_2018_over5_ = make_word_pos_list(tb_soon_2018_over5)

ebs_soon_2022_over5_ = make_word_pos_list(ebs_soon_2022_over5)
ebs_soon_2021_over5_ = make_word_pos_list(ebs_soon_2021_over5)
ebs_soon_2020_over5_ = make_word_pos_list(ebs_soon_2020_over5)
ebs_soon_2019_over5_ = make_word_pos_list(ebs_soon_2019_over5)
ebs_soon_2018_over5_ = make_word_pos_list(ebs_soon_2018_over5)

freq_tb_2022 = {word:freq for word, freq in textbook_2015curri_vocab.items() if word in tb_soon_2022_}
freq_tb_2021 = {word:freq for word, freq in textbook_2015curri_vocab.items() if word in tb_soon_2021_}
freq_tb_2020 = {word:freq for word, freq in textbook_2015curri_vocab.items() if word in tb_soon_2020_}
freq_tb_2019 = {word:freq for word, freq in textbook_2015curri_vocab.items() if word in tb_soon_2019_}
freq_tb_2018 = {word:freq for word, freq in textbook_2015curri_vocab.items() if word in tb_soon_2018_}

freq_ebs_2022 = {word:freq for word, freq in ebs_2022_vocab.items() if word in ebs_soon_2022_}
freq_ebs_2021 = {word:freq for word, freq in ebs_2021_vocab.items() if word in ebs_soon_2021_}
freq_ebs_2020 = {word:freq for word, freq in ebs_2020_vocab.items() if word in ebs_soon_2020_}
freq_ebs_2019 = {word:freq for word, freq in ebs_2019_vocab.items() if word in ebs_soon_2019_}
freq_ebs_2018 = {word:freq for word, freq in ebs_2018_vocab.items() if word in ebs_soon_2018_}

import pandas as pd
df_tb_freq = pd.DataFrame({'word': [kiwi.join(x) for x,y in freq_tb_2022.items()],
                           'freq': [y for x,y in freq_tb_2022.items()]})

from pandasgui import show
import os
os.environ['APPDATA'] = ""
show(df_tb_freq)