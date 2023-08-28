'''
After picklizing on main.py, I import pickle files
and compare each files for their words.
Consequently, overlapped words and their frequencies are calculated.

'''
# ***********************************************************************
# SOONONG WORDS
# ***********************************************************************
import copy
import os
import pickle
from pandas import DataFrame

# 수능 텍스트에서 추출한 어휘를 불러옵니다.
# 어휘는 빈도가 1 이상으로 나타난 어휘 모두를 추출한 결과입니다.

pfn_ksat =[
    '2018_ksat_vocab.pickle',
    '2019_ksat_vocab.pickle',
    '2020_ksat_vocab.pickle',
    '2021_ksat_vocab.pickle',
    '2022_ksat_vocab.pickle',
]

pickle_folder_path = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/5. pickles/ksat'

# import pickle files
# 2018학년도 수능
pickle_file_path = os.path.join(pickle_folder_path, pfn_ksat[0])
with open(pickle_file_path, 'rb') as pf:
    soonong_2018_vocab = pickle.load(pf)

# 2019학년도 수능
pickle_file_path = os.path.join(pickle_folder_path, pfn_ksat[1])
with open(pickle_file_path, 'rb') as pf:
    soonong_2019_vocab = pickle.load(pf)

# 2020학년도 수능
pickle_file_path = os.path.join(pickle_folder_path, pfn_ksat[2])
with open(pickle_file_path, 'rb') as pf:
    soonong_2020_vocab = pickle.load(pf)

# 2021학년도 수능
pickle_file_path = os.path.join(pickle_folder_path, pfn_ksat[3])
with open(pickle_file_path, 'rb') as pf:
    soonong_2021_vocab = pickle.load(pf)

# 2022학년도 수능
pickle_file_path = os.path.join(pickle_folder_path, pfn_ksat[4])
with open(pickle_file_path, 'rb') as pf:
    soonong_2022_vocab = pickle.load(pf)

# ***********************************************************************
# EBS WORDS
# ***********************************************************************
from dict_tools import dict_merger, dict_over_freq_filter

pfn_ebss = [
    '2018_ebs_vocab.pickle',
    '2019_ebs_vocab.pickle',
    '2020_ebs_vocab.pickle',
    '2021_ebs_vocab.pickle',
    '2022_ebs_vocab.pickle',
]

pickle_folder_path = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/5. pickles/ebs'

# import pickle files
# 2018학년도 EBS
pickle_file_path = os.path.join(pickle_folder_path, pfn_ebss[0])
with open(pickle_file_path, 'rb') as pf:
    ebs_2018_vocab = pickle.load(pf)

# 2019학년도 EBS
pickle_file_path = os.path.join(pickle_folder_path, pfn_ebss[1])
with open(pickle_file_path, 'rb') as pf:
    ebs_2019_vocab = pickle.load(pf)

# 2020학년도 EBS
pickle_file_path = os.path.join(pickle_folder_path, pfn_ebss[2])
with open(pickle_file_path, 'rb') as pf:
    ebs_2020_vocab = pickle.load(pf)

# 2021학년도 EBS
pickle_file_path = os.path.join(pickle_folder_path, pfn_ebss[3])
with open(pickle_file_path, 'rb') as pf:
    ebs_2021_vocab = pickle.load(pf)

# 2022학년도 EBS
pickle_file_path = os.path.join(pickle_folder_path, pfn_ebss[4])
with open(pickle_file_path, 'rb') as pf:
    ebs_2022_vocab = pickle.load(pf)


# 전체 freq 보존
ebs_2022_vocab_all = copy.deepcopy(ebs_2022_vocab)
ebs_2021_vocab_all = copy.deepcopy(ebs_2021_vocab)
ebs_2020_vocab_all = copy.deepcopy(ebs_2020_vocab)
ebs_2019_vocab_all = copy.deepcopy(ebs_2019_vocab)
ebs_2018_vocab_all = copy.deepcopy(ebs_2018_vocab)
len(ebs_2022_vocab_all)
len(ebs_2022_vocab)

ebs_2022_vocab


# ***********************************************************************
# TEXTBOOK WORDS
# ***********************************************************************
import pandas as pd

pickle_tb_file_path = '/Users/kintch/Dropbox/sj/2022-2/4. 수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10:3마감/textbook_pickles/tb_all_vocab.pickle'
with open(pickle_tb_file_path, 'rb') as pf:
    textbook_2015curri_vocab = pickle.load(pf)

# 전체 freq 보존
textbook_2015curri_vocab_all = copy.deepcopy(textbook_2015curri_vocab)




# 오류 교정
## 교과서
# (예): ('//NNG',)
def edit_pos_error(input_dict):
    tmp = dict()
    for x, freq in input_dict.items():
        if '//NNG' in str(x):
            print('발견!!', str(x))
        elif '/가/NNG' in str(x):
            print('발견!!', str(x))
        elif '/가외/NNG' in str(x):
            print('발견!!', str(x))
        else:
            tmp[x] = freq
    return tmp


# 형태소 분석 오류 교정
# (예) ('//NNG',)
# 교과서 어휘
textbook_2015curri_vocab_ = edit_pos_error(textbook_2015curri_vocab)
textbook_2015curri_vocab_all_ = edit_pos_error(textbook_2015curri_vocab_all)

# EBS 어휘
ebs_2022_vocab_all_ = edit_pos_error(ebs_2022_vocab_all)
ebs_2021_vocab_all_ = edit_pos_error(ebs_2021_vocab_all)
ebs_2020_vocab_all_ = edit_pos_error(ebs_2020_vocab_all)
ebs_2019_vocab_all_ = edit_pos_error(ebs_2019_vocab_all)
ebs_2018_vocab_all_ = edit_pos_error(ebs_2018_vocab_all)

ebs_2022_vocab_ = edit_pos_error(ebs_2022_vocab)
ebs_2021_vocab_ = edit_pos_error(ebs_2021_vocab)
ebs_2020_vocab_ = edit_pos_error(ebs_2020_vocab)
ebs_2019_vocab_ = edit_pos_error(ebs_2019_vocab)
ebs_2018_vocab_ = edit_pos_error(ebs_2018_vocab)

soonong_2022_vocab_ = edit_pos_error(soonong_2022_vocab)
soonong_2021_vocab_ = edit_pos_error(soonong_2021_vocab)
soonong_2020_vocab_ = edit_pos_error(soonong_2020_vocab)
soonong_2019_vocab_ = edit_pos_error(soonong_2019_vocab)
soonong_2018_vocab_ = edit_pos_error(soonong_2018_vocab)



# 오류 교정 후 복사
textbook_2015curri_vocab = copy.deepcopy(textbook_2015curri_vocab_)
textbook_2015curri_vocab_all = copy.deepcopy(textbook_2015curri_vocab_all_)

ebs_2022_vocab_all = copy.deepcopy(ebs_2022_vocab_all_)
ebs_2021_vocab_all = copy.deepcopy(ebs_2021_vocab_all_)
ebs_2020_vocab_all = copy.deepcopy(ebs_2020_vocab_all_)
ebs_2019_vocab_all = copy.deepcopy(ebs_2019_vocab_all_)
ebs_2018_vocab_all = copy.deepcopy(ebs_2018_vocab_all_)

ebs_2022_vocab = copy.deepcopy(ebs_2022_vocab_)
ebs_2021_vocab = copy.deepcopy(ebs_2021_vocab_)
ebs_2020_vocab = copy.deepcopy(ebs_2020_vocab_)
ebs_2019_vocab = copy.deepcopy(ebs_2019_vocab_)
ebs_2018_vocab = copy.deepcopy(ebs_2018_vocab_)

soonong_2022_vocab = copy.deepcopy(soonong_2022_vocab_)
soonong_2021_vocab = copy.deepcopy(soonong_2021_vocab_)
soonong_2020_vocab = copy.deepcopy(soonong_2020_vocab_)
soonong_2019_vocab = copy.deepcopy(soonong_2019_vocab_)
soonong_2018_vocab = copy.deepcopy(soonong_2018_vocab_)

# 기존 삭제

del textbook_2015curri_vocab_
del textbook_2015curri_vocab_all_

del ebs_2022_vocab_all_
del ebs_2021_vocab_all_
del ebs_2020_vocab_all_
del ebs_2019_vocab_all_
del ebs_2018_vocab_all_

del ebs_2022_vocab_
del ebs_2021_vocab_
del ebs_2020_vocab_
del ebs_2019_vocab_
del ebs_2018_vocab_

del soonong_2022_vocab_
del soonong_2021_vocab_
del soonong_2020_vocab_
del soonong_2019_vocab_
del soonong_2018_vocab_



# pd.DataFrame.from_dict(textbook_2015curri_vocab, orient='index')


# ***********************************************************************
# DEFINITION
# ***********************************************************************

# (친숙하지 않은 어휘) = (수능 어휘) - (EBS에서 추출한 친숙한 어휘) - (교과서에서 추출한 친숙한 어휘)
# (e.g.) 2022학년도 수능 비문학의 친숙하지 않은 어휘 = (2022학년도 수능 비문학에서 등장한 어휘) - (2022학년도 EBS에서 추출한 친숙한 어휘) - (교과서에서 추출한 친숙한 어휘)

# 교과서의 친숙한 어휘: textbook_2015curri_vocab (type: dict)
# EBS의 친숙한 어휘: ebs_2022_vocab ~ ebs_2018_vocab (type: dict)
# 수능 어휘: soonong_2022_vocab ~ soonong_2018_vocab (type: dict)


# ***********************************************************************
# RESULT(demo)
# ***********************************************************************

def restore_word_from_pos(possed_token):
    #keyw == possed_token
    # restore word from tokenized results
    # before_join = [tuple(x.split(r'/')) for x in possed_token]
    # print(before_join)
    restored_word = kiwi.join(possed_token)
    return restored_word

# restore_word_from_pos(('아버지/NNG', '의/JKG', '성격/NNG'))
# re딕셔너링
# ('아버지/NNG', '의/JKG', '성격/NNG') --> 아버지의성격
# (('아버지, 'NNG'), ('의, 'JKG'), ('성격, 'NNG')) --> 아버지의성격

import re
def redictionaring_from_poskeyw(input_dict):
    tmp = dict()
    for keyw, freq in input_dict.items():
        edited_keyw = re.sub(' ', '', restore_word_from_pos(keyw))  #띄어쓰기 제거
        if edited_keyw in tmp:
            print('이미 있었네!', edited_keyw, 'from: ', keyw)
            tmp[edited_keyw] = tmp[edited_keyw] + freq
        else:
            # print(keyw, edited_keyw)
            tmp[edited_keyw] = freq
    return tmp

## 교과서
# 형태소 -> 단어로 만드는 과정에서 근로+기준+법 -> 근로+기준법 과 같은 단어들이 한 단어로 재처리됨.

textbook_2015curri_vocab_ = redictionaring_from_poskeyw(textbook_2015curri_vocab)
textbook_2015curri_vocab_all_ = redictionaring_from_poskeyw(textbook_2015curri_vocab_all)
## 수능
soonong_2022_vocab_ = redictionaring_from_poskeyw(soonong_2022_vocab)
soonong_2021_vocab_ = redictionaring_from_poskeyw(soonong_2021_vocab)
soonong_2020_vocab_ = redictionaring_from_poskeyw(soonong_2020_vocab)
soonong_2019_vocab_ = redictionaring_from_poskeyw(soonong_2019_vocab)
soonong_2018_vocab_ = redictionaring_from_poskeyw(soonong_2018_vocab)
## EBS
ebs_2022_vocab_ = redictionaring_from_poskeyw(ebs_2022_vocab)
ebs_2021_vocab_ = redictionaring_from_poskeyw(ebs_2021_vocab)
ebs_2020_vocab_ = redictionaring_from_poskeyw(ebs_2020_vocab)
ebs_2019_vocab_ = redictionaring_from_poskeyw(ebs_2019_vocab)
ebs_2018_vocab_ = redictionaring_from_poskeyw(ebs_2018_vocab)

ebs_2022_vocab_all_ = redictionaring_from_poskeyw(ebs_2022_vocab_all)
ebs_2021_vocab_all_ = redictionaring_from_poskeyw(ebs_2021_vocab_all)
ebs_2020_vocab_all_ = redictionaring_from_poskeyw(ebs_2020_vocab_all)
ebs_2019_vocab_all_ = redictionaring_from_poskeyw(ebs_2019_vocab_all)
ebs_2018_vocab_all_ = redictionaring_from_poskeyw(ebs_2018_vocab_all)

# 이제 all과의 차이를 만든다.
# EBS는 빈도 5 이상 필터링
ebs_2022_vocab_ = dict_over_freq_filter(ebs_2022_vocab_, over_freq=5)
ebs_2021_vocab_ = dict_over_freq_filter(ebs_2021_vocab_, over_freq=5)
ebs_2020_vocab_ = dict_over_freq_filter(ebs_2020_vocab_, over_freq=5)
ebs_2019_vocab_ = dict_over_freq_filter(ebs_2019_vocab_, over_freq=5)
ebs_2018_vocab_ = dict_over_freq_filter(ebs_2018_vocab_, over_freq=5)
# 교과서 빈도 5이상 필터링
textbook_2015curri_vocab_ = dict_over_freq_filter(textbook_2015curri_vocab_, over_freq=5)

# 어휘 리스트 뽑기
### 2022
# 2022학년도 수능 비문학에 등장한 어휘 리스트
soonong_2022_vocab_list = list()
for keyw, freq in soonong_2022_vocab_.items():
    soonong_2022_vocab_list.append(keyw)
# 2022학년도 EBS에서 추출한 친숙한 어휘 리스트
ebs_2022_vocab_list = list()
for keyw, freq in ebs_2022_vocab_.items():
    ebs_2022_vocab_list.append(keyw)

### 2021
# 2021학년도 수능 비문학에 등장한 어휘 리스트
soonong_2021_vocab_list = list()
for keyw, freq in soonong_2021_vocab_.items():
    soonong_2021_vocab_list.append(keyw)
# 2021학년도 EBS에서 추출한 친숙한 어휘 리스트
ebs_2021_vocab_list = list()
for keyw, freq in ebs_2021_vocab_.items():
    ebs_2021_vocab_list.append(keyw)

### 2020
# 2020학년도 수능 비문학에 등장한 어휘 리스트
soonong_2020_vocab_list = list()
for keyw, freq in soonong_2020_vocab_.items():
    soonong_2020_vocab_list.append(keyw)
# 2020학년도 EBS에서 추출한 친숙한 어휘 리스트
ebs_2020_vocab_list = list()
for keyw, freq in ebs_2020_vocab_.items():
    ebs_2020_vocab_list.append(keyw)

### 2019
# 2019학년도 수능 비문학에 등장한 어휘 리스트
soonong_2019_vocab_list = list()
for keyw, freq in soonong_2019_vocab_.items():
    soonong_2019_vocab_list.append(keyw)
# 2019학년도 EBS에서 추출한 친숙한 어휘 리스트
ebs_2019_vocab_list = list()
for keyw, freq in ebs_2019_vocab_.items():
    ebs_2019_vocab_list.append(keyw)

### 2018
# 2018학년도 수능 비문학에 등장한 어휘 리스트
soonong_2018_vocab_list = list()
for keyw, freq in soonong_2018_vocab_.items():
    soonong_2018_vocab_list.append(keyw)
# 2018학년도 EBS에서 추출한 친숙한 어휘 리스트
ebs_2018_vocab_list = list()
for keyw, freq in ebs_2018_vocab_.items():
    ebs_2018_vocab_list.append(keyw)



### 대망의
# 2015개정 교과서에서 추출한 친숙한 어휘 리스트
textbook_2015curri_vocab_list = list()
for keyw, freq in textbook_2015curri_vocab_.items():
    # print('keyw:', keyw)
    textbook_2015curri_vocab_list.append(keyw)



# 전체 어휘 리스트
import numpy as np
soonong_2022_vocab_list
ave_soonong_vocab_num = np.average([len(soonong_2022_vocab_list),
len(soonong_2021_vocab_list),
len(soonong_2020_vocab_list),
len(soonong_2019_vocab_list),
len(soonong_2018_vocab_list)])
ave_soonong_vocab_num

std_soonong_vocab_num = np.std([len(soonong_2022_vocab_list),
len(soonong_2021_vocab_list),
len(soonong_2020_vocab_list),
len(soonong_2019_vocab_list),
len(soonong_2018_vocab_list)])
std_soonong_vocab_num



## 교과서 연계(2018~2022)
textbook_based_2022_soonong_words = list(set(soonong_2022_vocab_list) & set(textbook_2015curri_vocab_list))
textbook_based_2021_soonong_words = list(set(soonong_2021_vocab_list) & set(textbook_2015curri_vocab_list))
textbook_based_2020_soonong_words = list(set(soonong_2020_vocab_list) & set(textbook_2015curri_vocab_list))
textbook_based_2019_soonong_words = list(set(soonong_2019_vocab_list) & set(textbook_2015curri_vocab_list))
textbook_based_2018_soonong_words = list(set(soonong_2018_vocab_list) & set(textbook_2015curri_vocab_list))


tb_soonong_words = (set(textbook_based_2022_soonong_words) |
 set(textbook_based_2021_soonong_words) |
 set(textbook_based_2020_soonong_words) |
 set(textbook_based_2019_soonong_words) |
 set(textbook_based_2018_soonong_words))


ave_tb_soon_vocab_num = np.average([len(textbook_based_2022_soonong_words),
                                    len(textbook_based_2021_soonong_words),
                                    len(textbook_based_2020_soonong_words),
                                    len(textbook_based_2019_soonong_words),
                                    len(textbook_based_2018_soonong_words)])

std_tb_soon_vocab_num = np.std([len(textbook_based_2022_soonong_words),
                                len(textbook_based_2021_soonong_words),
                                len(textbook_based_2020_soonong_words),
                                len(textbook_based_2019_soonong_words),
                                len(textbook_based_2018_soonong_words)])
std_tb_soon_vocab_num

ave_tb_related_percent = (len(textbook_based_2022_soonong_words)*100/len(soonong_2022_vocab_list) + \
len(textbook_based_2021_soonong_words)*100/len(soonong_2021_vocab_list) + \
len(textbook_based_2020_soonong_words)*100/len(soonong_2020_vocab_list) + \
len(textbook_based_2019_soonong_words)*100/len(soonong_2019_vocab_list) + \
len(textbook_based_2018_soonong_words)*100/len(soonong_2018_vocab_list))/5

ave_tb_related_percent_2022 = len(textbook_based_2022_soonong_words)*100/len(soonong_2022_vocab_list)
ave_tb_related_percent_2021 = len(textbook_based_2021_soonong_words)*100/len(soonong_2021_vocab_list)
ave_tb_related_percent_2020 = len(textbook_based_2020_soonong_words)*100/len(soonong_2020_vocab_list)
ave_tb_related_percent_2019 = len(textbook_based_2019_soonong_words)*100/len(soonong_2019_vocab_list)
ave_tb_related_percent_2018 = len(textbook_based_2018_soonong_words)*100/len(soonong_2018_vocab_list)



## EBS 연계
ebs_based_2022_soonong_words = list(set(soonong_2022_vocab_list) & set(ebs_2022_vocab_list))
ebs_based_2021_soonong_words = list(set(soonong_2021_vocab_list) & set(ebs_2021_vocab_list))
ebs_based_2020_soonong_words = list(set(soonong_2020_vocab_list) & set(ebs_2020_vocab_list))
ebs_based_2019_soonong_words = list(set(soonong_2019_vocab_list) & set(ebs_2019_vocab_list))
ebs_based_2018_soonong_words = list(set(soonong_2018_vocab_list) & set(ebs_2018_vocab_list))

ave_ebs_soon_vocab_num = np.average([len(ebs_based_2022_soonong_words),
                                     len(ebs_based_2021_soonong_words),
                                     len(ebs_based_2020_soonong_words),
                                     len(ebs_based_2019_soonong_words),
                                     len(ebs_based_2018_soonong_words)])
std_ebs_soon_vocab_num = np.std([len(ebs_based_2022_soonong_words),
                                 len(ebs_based_2021_soonong_words),
                                 len(ebs_based_2020_soonong_words),
                                 len(ebs_based_2019_soonong_words),
                                 len(ebs_based_2018_soonong_words)])
ave_ebs_soon_vocab_num
std_ebs_soon_vocab_num

ebs_soonong_words = (set(ebs_based_2022_soonong_words) |
set(ebs_based_2021_soonong_words) |
set(ebs_based_2020_soonong_words) |
set(ebs_based_2019_soonong_words) |
set(ebs_based_2018_soonong_words))



ave_ebs_related_percent = (len(ebs_based_2022_soonong_words)*100/len(soonong_2022_vocab_list) + \
len(ebs_based_2021_soonong_words)*100/len(soonong_2021_vocab_list) + \
len(ebs_based_2020_soonong_words)*100/len(soonong_2020_vocab_list) + \
len(ebs_based_2019_soonong_words)*100/len(soonong_2019_vocab_list) + \
len(ebs_based_2018_soonong_words)*100/len(soonong_2018_vocab_list))/5


def overlap_inlist_finder(input_list):
    if len(input_list) != len(list(set(input_list))):
        print('overlapped!')

overlap_inlist_finder(soonong_2022_vocab_list)
overlap_inlist_finder(soonong_2021_vocab_list)
overlap_inlist_finder(soonong_2020_vocab_list)
overlap_inlist_finder(soonong_2019_vocab_list)
overlap_inlist_finder(soonong_2018_vocab_list)
overlap_inlist_finder(ebs_2022_vocab_list)
overlap_inlist_finder(ebs_2021_vocab_list)
overlap_inlist_finder(ebs_2020_vocab_list)
overlap_inlist_finder(ebs_2019_vocab_list)
overlap_inlist_finder(ebs_2018_vocab_list)
overlap_inlist_finder(textbook_2015curri_vocab_list)


## 미 연계
unfam_2022_soonong_words = list(set(soonong_2022_vocab_list) - set(ebs_2022_vocab_list) - set(textbook_2015curri_vocab_list))
unfam_2021_soonong_words = list(set(soonong_2021_vocab_list) - set(ebs_2021_vocab_list) - set(textbook_2015curri_vocab_list))
unfam_2020_soonong_words = list(set(soonong_2020_vocab_list) - set(ebs_2020_vocab_list) - set(textbook_2015curri_vocab_list))
unfam_2019_soonong_words = list(set(soonong_2019_vocab_list) - set(ebs_2019_vocab_list) - set(textbook_2015curri_vocab_list))
unfam_2018_soonong_words = list(set(soonong_2018_vocab_list) - set(ebs_2018_vocab_list) - set(textbook_2015curri_vocab_list))

unfam_soonong_words = (set(unfam_2022_soonong_words) |
set(unfam_2021_soonong_words) |
set(unfam_2020_soonong_words) |
set(unfam_2019_soonong_words) |
set(unfam_2018_soonong_words))


tb_soonong_words_ = list(tb_soonong_words)
ebs_soonong_words_ = list(ebs_soonong_words)
unfam_soonong_words_ = list(unfam_soonong_words)

set([1,2]) & set([1,3,4,5]) & set([1,2,3])
set([1,2,3])-set([1,2])-set([1,3,4,5])
len(unfam_2022_soonong_words)
len(soonong_2022_vocab_list)
len(textbook_based_2022_soonong_words)
len(ebs_based_2022_soonong_words)
len(textbook_n_ebs_based_2022_soonong_words)




[len(unfam_2022_soonong_words),
len(unfam_2021_soonong_words),
len(unfam_2020_soonong_words),
len(unfam_2019_soonong_words),
len(unfam_2018_soonong_words)]

ave_un_related_percent = np.average([len(unfam_2022_soonong_words)*100/len(soonong_2022_vocab_list),
len(unfam_2021_soonong_words)*100/len(soonong_2021_vocab_list),
len(unfam_2020_soonong_words)*100/len(soonong_2020_vocab_list),
len(unfam_2019_soonong_words)*100/len(soonong_2019_vocab_list),
len(unfam_2018_soonong_words)*100/len(soonong_2018_vocab_list)])



textbook_n_ebs_based_2022_soonong_words = list(set(soonong_2022_vocab_list) & set(textbook_2015curri_vocab_list) & set(ebs_2022_vocab_list))
textbook_n_ebs_based_2021_soonong_words = list(set(soonong_2021_vocab_list) & set(textbook_2015curri_vocab_list) & set(ebs_2021_vocab_list))
textbook_n_ebs_based_2020_soonong_words = list(set(soonong_2020_vocab_list) & set(textbook_2015curri_vocab_list) & set(ebs_2020_vocab_list))
textbook_n_ebs_based_2019_soonong_words = list(set(soonong_2019_vocab_list) & set(textbook_2015curri_vocab_list) & set(ebs_2019_vocab_list))
textbook_n_ebs_based_2018_soonong_words = list(set(soonong_2018_vocab_list) & set(textbook_2015curri_vocab_list) & set(ebs_2018_vocab_list))
[len(textbook_n_ebs_based_2022_soonong_words),
 len(textbook_n_ebs_based_2021_soonong_words),
 len(textbook_n_ebs_based_2020_soonong_words),
 len(textbook_n_ebs_based_2019_soonong_words),
 len(textbook_n_ebs_based_2018_soonong_words)]

# 연계/미연계 어휘 딕셔너리
# 딕셔너리 나중에 합쳐야 해
def dict_merger(dict1, dict2):
    tmp = dict()
    for x, y in dict1.items():
        tmp[x] = y
    # print('tmp', tmp)
    for x2, y2 in dict2.items():
        if x2 in tmp:
            # print(x2, 'yes')
            tmp[x2] = tmp[x2] + y2
        else:
            tmp[x2] = y2
    return tmp

## 교과서 연계 어휘의 교과서 내 빈도
textbook_based_2022_soonong_words_dict = dict()
for keyw in textbook_based_2022_soonong_words:
    keyw_freq = textbook_2015curri_vocab_all_[keyw]
    textbook_based_2022_soonong_words_dict[keyw] = keyw_freq

textbook_based_2021_soonong_words_dict = dict()
for keyw in textbook_based_2021_soonong_words:
    keyw_freq = textbook_2015curri_vocab_all_[keyw]
    textbook_based_2021_soonong_words_dict[keyw] = keyw_freq

textbook_based_2020_soonong_words_dict = dict()
for keyw in textbook_based_2020_soonong_words:
    keyw_freq = textbook_2015curri_vocab_all_[keyw]
    textbook_based_2020_soonong_words_dict[keyw] = keyw_freq

textbook_based_2019_soonong_words_dict = dict()
for keyw in textbook_based_2019_soonong_words:
    keyw_freq = textbook_2015curri_vocab_all_[keyw]
    textbook_based_2019_soonong_words_dict[keyw] = keyw_freq

textbook_based_2018_soonong_words_dict = dict()
for keyw in textbook_based_2018_soonong_words:
    keyw_freq = textbook_2015curri_vocab_all_[keyw]
    textbook_based_2018_soonong_words_dict[keyw] = keyw_freq

tb_soon_all = {**textbook_based_2018_soonong_words_dict,
 **textbook_based_2019_soonong_words_dict,
 **textbook_based_2020_soonong_words_dict,
 **textbook_based_2021_soonong_words_dict,
 **textbook_based_2022_soonong_words_dict,
 }


## EBS 연계 어휘의 EBS 내 빈도
ebs_based_2022_soonong_words_dict = dict()
for keyw in ebs_based_2022_soonong_words:
    keyw_freq = ebs_2022_vocab_all_[keyw]
    ebs_based_2022_soonong_words_dict[keyw] = keyw_freq

ebs_based_2021_soonong_words_dict = dict()
for keyw in ebs_based_2021_soonong_words:
    keyw_freq = ebs_2021_vocab_all_[keyw]
    ebs_based_2021_soonong_words_dict[keyw] = keyw_freq

ebs_based_2020_soonong_words_dict = dict()
for keyw in ebs_based_2020_soonong_words:
    keyw_freq = ebs_2020_vocab_all_[keyw]
    ebs_based_2020_soonong_words_dict[keyw] = keyw_freq

ebs_based_2019_soonong_words_dict = dict()
for keyw in ebs_based_2019_soonong_words:
    keyw_freq = ebs_2019_vocab_all_[keyw]
    ebs_based_2019_soonong_words_dict[keyw] = keyw_freq

ebs_based_2018_soonong_words_dict = dict()
for keyw in ebs_based_2018_soonong_words:
    keyw_freq = ebs_2018_vocab_all_[keyw]
    ebs_based_2018_soonong_words_dict[keyw] = keyw_freq

ebs_soon_all = {**ebs_based_2018_soonong_words_dict,
 **ebs_based_2019_soonong_words_dict,
 **ebs_based_2020_soonong_words_dict,
 **ebs_based_2021_soonong_words_dict,
 **ebs_based_2022_soonong_words_dict,
 }



## 미 연계 어휘의 EBS/교과서 내 빈도
def cal_inebs_or_intb_freq(input_list, vocab_dict=ebs_2022_vocab_all_):
    tmp=list()
    for keyw in input_list:
        try:
            tmp.append(vocab_dict[keyw])
        except:
            tmp.append(0)
    return tmp

unfam_2022_soonong_words_inebs_freq = cal_inebs_or_intb_freq(unfam_2022_soonong_words, vocab_dict=ebs_2022_vocab_all_)
unfam_2021_soonong_words_inebs_freq = cal_inebs_or_intb_freq(unfam_2021_soonong_words, vocab_dict=ebs_2021_vocab_all_)
unfam_2020_soonong_words_inebs_freq = cal_inebs_or_intb_freq(unfam_2020_soonong_words, vocab_dict=ebs_2020_vocab_all_)
unfam_2019_soonong_words_inebs_freq = cal_inebs_or_intb_freq(unfam_2019_soonong_words, vocab_dict=ebs_2019_vocab_all_)
unfam_2018_soonong_words_inebs_freq = cal_inebs_or_intb_freq(unfam_2022_soonong_words, vocab_dict=ebs_2018_vocab_all_)

unfam_2022_soonong_words_intb_freq = cal_inebs_or_intb_freq(unfam_2022_soonong_words, vocab_dict=textbook_2015curri_vocab_all_)
unfam_2021_soonong_words_intb_freq = cal_inebs_or_intb_freq(unfam_2021_soonong_words, vocab_dict=textbook_2015curri_vocab_all_)
unfam_2020_soonong_words_intb_freq = cal_inebs_or_intb_freq(unfam_2020_soonong_words, vocab_dict=textbook_2015curri_vocab_all_)
unfam_2019_soonong_words_intb_freq = cal_inebs_or_intb_freq(unfam_2019_soonong_words, vocab_dict=textbook_2015curri_vocab_all_)
unfam_2018_soonong_words_intb_freq = cal_inebs_or_intb_freq(unfam_2018_soonong_words, vocab_dict=textbook_2015curri_vocab_all_)


np.average([np.average(unfam_2022_soonong_words_inebs_freq),
           np.average(unfam_2021_soonong_words_inebs_freq),
           np.average(unfam_2020_soonong_words_inebs_freq),
           np.average(unfam_2019_soonong_words_inebs_freq),
           np.average(unfam_2018_soonong_words_inebs_freq)])


np.average([np.average(unfam_2022_soonong_words_intb_freq),
           np.average(unfam_2021_soonong_words_intb_freq),
           np.average(unfam_2020_soonong_words_intb_freq),
           np.average(unfam_2019_soonong_words_intb_freq),
           np.average(unfam_2018_soonong_words_intb_freq)])

def ina_or_inb_checker(a, b):
    sum=0
    for x in range(len(a)):
        if a[x] != 0:
            sum+=1
        else:
            if b[x] != 0:
                sum += 1
    return sum/len(a)

ina_or_inb_checker(unfam_2022_soonong_words_inebs_freq, unfam_2022_soonong_words_intb_freq)
37*ina_or_inb_checker(unfam_2021_soonong_words_inebs_freq, unfam_2021_soonong_words_intb_freq)
34*ina_or_inb_checker(unfam_2020_soonong_words_inebs_freq, unfam_2020_soonong_words_intb_freq)
38*ina_or_inb_checker(unfam_2019_soonong_words_inebs_freq, unfam_2019_soonong_words_intb_freq)
42*ina_or_inb_checker(unfam_2018_soonong_words_inebs_freq, unfam_2018_soonong_words_intb_freq)


sum
42*110/377
# freq를 기반으로 하는 plot 뽑기
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
weight=unfam_2022_soonong_words_inebs_freq+\
unfam_2021_soonong_words_inebs_freq+\
unfam_2020_soonong_words_inebs_freq+\
unfam_2019_soonong_words_inebs_freq+\
unfam_2018_soonong_words_inebs_freq

weight2=unfam_2022_soonong_words_intb_freq+\
unfam_2021_soonong_words_intb_freq+\
unfam_2020_soonong_words_intb_freq+\
unfam_2019_soonong_words_intb_freq+\
unfam_2018_soonong_words_intb_freq

plt.hist(weight, color='green', bins=10, range=[0,10], density=True, alpha=0.8, label='EBS 내 빈도')
plt.xticks(list(range(0,11)))
plt.hist(weight2, color='red', bins=10, range=[0,10], density=True, alpha=0.6, label='교과서 내 빈도')
plt.legend()
plt.ylabel('ratio')
plt.xlabel('frequency')
plt.show()

x = np.arange(3)
years = ['2018', '2019', '2020']
values = [100, 400, 900]

plt.bar(x, values)
plt.xticks(x, years)
plt.show()

import pandas as pd
pd.DataFrame.from_dict(unfam_2022_soonong_words_dict, orient='index')


sns.countplot(x="class", hue="who", palette='Paired', data=DataFrame.from_dict(unfam_2022_soonong_words_dict))
plt.show()





# dataframe to csv 뽑기

# ***********************************************************************
# ReFiltering (템플릿 기반 필터링)
# ***********************************************************************
from vocab_tools import templated_ngram_dict_filter

neg_template_list = [
    # 부호, 외국어, 특수문자로 시작 ## 왜 DNA가 없을까?
    ('/S',),
    tuple(['/S'] + [''] * 1),
    tuple(['/S'] + [''] * 2),
    tuple(['/S'] + [''] * 3),
    tuple(['/S'] + [''] * 4),

    # 조사로 시작
    ('/J',),
    tuple(['/J'] + [''] * 1),
    tuple(['/J'] + [''] * 2),
    tuple(['/J'] + [''] * 3),
    tuple(['/J'] + [''] * 4),

    # 부사로 시작
    ('/MA',),
    tuple(['/MA'] + [''] * 1),
    tuple(['/MA'] + [''] * 2),
    tuple(['/MA'] + [''] * 3),
    tuple(['/MA'] + [''] * 4),

    # 어미로 시작
    ('/E',),
    tuple(['/E'] + [''] * 1),
    tuple(['/E'] + [''] * 2),
    tuple(['/E'] + [''] * 3),
    tuple(['/E'] + [''] * 4),

    # 접두사, 접미사로 시작
    ('/X',),
    tuple(['/X'] + [''] * 1),
    tuple(['/X'] + [''] * 2),
    tuple(['/X'] + [''] * 3),
    tuple(['/X'] + [''] * 4),

    # 조사로 종료
    tuple([''] * 1 + ['/J']),
    tuple([''] * 2 + ['/J']),
    tuple([''] * 3 + ['/J']),
    tuple([''] * 4 + ['/J']),


    # 어미로 종료
    tuple([''] * 1 + ['/E']),
    tuple([''] * 2 + ['/E']),
    tuple([''] * 3 + ['/E']),
    tuple([''] * 4 + ['/E'])

]

neg_template_list

filtered_2022_results = templated_ngram_dict_filter(unfam_2022_soonong_words_dict,
                                                    neg_templates=neg_template_list)

filtered_2022_results

# ***********************************************************************
# Sort & Print (빈도순으로 정리 후 .txt로 프린팅)
# ***********************************************************************

# 빈도 기준으로 한 번 더 소팅
sorted_by_freq_list = sorted(filtered_2022_results.items(), key=lambda x: x[1], reverse=True)

# POS 템플릿 기준으로 소팅하는 함수 정의
import re


def sort_by_template(dict):
    template_indexed_word_list = list()
    for keyw, freq in dict:
        pos_template = tuple([re.sub(r'.*/', '', x) for x in keyw])
        template_indexed_word_list.append(tuple([keyw, freq, pos_template]))

    word_list_sorted_based_template = sorted(template_indexed_word_list, key=lambda x: x[2], reverse=True)
    return word_list_sorted_based_template


# POS 템플릿 기준으로 소팅
sorted_by_freq_n_temp_list = sort_by_template(sorted_by_freq_list)

# len(sorted_by_temp_n_freq_list)

# set_key_len = list()
# for keyw, freq in sorted_unfam_2022_soonong_words_list:
#     if len(keyw) == 1:
#         print(keyw)

# make
# key_len_list = list(set([len(keyw) for keyw, freq in sorted_unfam_2022_soonong_words_list]))

# save to .TXT
with open(
        '/Users/kintch/Dropbox/sj/2022-2/4. 수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10:3마감/unfam_vocab_2022.txt',
        'w', encoding='utf-8') as f:
    keyw_1 = str()
    keyw_2 = str()
    keyw_3 = str()
    keyw_4 = str()
    for keyw, freq, temp in sorted_by_freq_n_temp_list:
        if len(keyw) == 1:
            keyw_1 += str(temp) + '----' + str(keyw) + '/' + str(freq) + '\n'

        elif len(keyw) == 2:
            keyw_2 += str(temp) + '----' + str(keyw) + '/' + str(freq) + '\n'

        elif len(keyw) == 3:
            keyw_3 += str(temp) + '----' + str(keyw) + '/' + str(freq) + '\n'

        elif len(keyw) == 4:
            keyw_4 += str(temp) + '----' + str(keyw) + '/' + str(freq) + '\n'
    f.write(str("<한 단어 목록>") + '\n' + keyw_1 + '\n' + '\n' + '\n' +
            str("<두 단어 목록>") + '\n' + keyw_2 + '\n' + '\n' + '\n' +
            str("<세 단어 목록>") + '\n' + keyw_3 + '\n' + '\n' + '\n' +
            str("<네 단어 목록>") + '\n' + keyw_4 + '\n' + '\n' + '\n'
            )

# save to .CSV
import copy


def make_list_same_length(list, maxlen=4):
    return_list = copy.deepcopy(list)
    while len(return_list) < maxlen:
        return_list += ['']
    return return_list


import pandas as pd
from kiwipiepy import Kiwi

kiwi = Kiwi(num_workers=4, model_type='sbg')

key_freq_temp_frame = pd.DataFrame(
    columns=['keylen', 'restored_word', 'temp_sum', 'temp1', 'temp2', 'temp3', 'temp4', 'keyw', 'freq'])
for keyw, freq, temp in sorted_by_freq_n_temp_list:
    # restore word from tokenized results
    before_join = [tuple(x.split(r'/')) for x in keyw]
    restored_word = kiwi.join(before_join)

    # make pos list
    pos_list = make_list_same_length([x for x in temp])

    # def new row (will be added)
    new_row = pd.DataFrame([[len(keyw)] + [restored_word] + [temp] + pos_list + [keyw] + [freq]],
                           columns=key_freq_temp_frame.columns)

    # ADD
    key_freq_temp_frame = pd.concat([key_freq_temp_frame, new_row], ignore_index=True)


# key_freq_temp_frame.columns

key_freq_temp_frame = key_freq_temp_frame.sort_values('temp_sum').sort_values('keylen')
key_freq_temp_frame.to_excel('/Users/kintch/Dropbox/sj/2022-2/4. 수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10:3마감/unfam_vocab_2022.xlsx',
                             encoding='utf8')
