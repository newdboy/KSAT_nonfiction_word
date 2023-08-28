import os
import pickle
from dict_tools import dict_merger, dict_over_freq_filter
import pprint
'''
교과서 pickle 파일을 불러다가 친숙한 어휘의 목록을 만듭니다.
친숙한 어휘는 각 교과마다 빈도가 k 이상으로 나타난 어휘이며,
과목별로 사용한 교과서의 출판사의 수를 감안하여, normalization을 해준 결과에 따라 계산된 것입니다. 
'''

# 폴더명 5.pickles 뒤에 '_' 붙였음. (실수로 바뀌지 않게)
pickle_path = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/5. pickles/textbook'
# Make file path (folder path & file name) list
pickle_path_list = list()
for (root, directories, files) in os.walk(pickle_path):
    pickle_path_list.append([root, files])  # [루트, [파일명1, 파일명2 ...]] (list)을 추출


# sorted(pickle_path_list[0][1])
pickle_file_names = [
'00.공통국어_textbook_vocab.pickle',
 '01.독서_textbook_vocab.pickle',
 '02.문학_textbook_vocab.pickle',
 '03.화법과작문_textbook_vocab.pickle',
 '04.언어와매체_textbook_vocab.pickle',
 '05.통합사회_textbook_vocab.pickle',
 '06.사회문화_textbook_vocab.pickle',
 '07.한국지리_textbook_vocab.pickle',
 '08.세계지리_textbook_vocab.pickle',
 '09.윤리와사상_textbook_vocab.pickle',
 '10.생활과 윤리_textbook_vocab.pickle',
 '11.통합과학_textbook_vocab.pickle',
 '12.물리학1_textbook_vocab.pickle',
 '13.화학1_textbook_vocab.pickle',
 '14.생명과학1_textbook_vocab.pickle',
 '15.지구과학1_textbook_vocab.pickle'
]


number_of_textbooks = [
    11,
    6,
    9,
    5,
    5,
    5,
    5,
    3,
    4,
    5,
    5,
    5,
    8,
    8,
    6,
    9
]

len(number_of_textbooks)
# merge all pickle files (with weighted freq)
all_tb_vocab_dict = dict()
file_num = 0
for pfn in pickle_file_names:

    # testin
    # file_num = 8
    # pfn = pickle_file_names[8]
    print(pfn)

    # import pickle file
    pickle_file_path = os.path.join(pickle_path, pfn)
    with open (pickle_file_path, 'rb') as pf:
        data = pickle.load(pf)

    # weighted freq
    data_ = dict()
    for word, freq in data.items():
        data_[word] = freq / number_of_textbooks[file_num]
    print('file_name:', pfn, ', weight:', number_of_textbooks[file_num])  # pickle_file_name

    # merge weighted dict (with all_tb_vocab_dict)
    all_tb_vocab_dict = dict_merger(all_tb_vocab_dict, data_)

    file_num += 1


# 빈도 1 이상 필터링**
filtered_tb_vocab_dict = dict_over_freq_filter(all_tb_vocab_dict, over_freq=1)

# pickle로 저장

pickle_tb_file_dir = \
    '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/5. pickles/textbook/tb_all_vocab.pickle'
with open(pickle_tb_file_dir, 'wb') as wp:
    pickle.dump(filtered_tb_vocab_dict, wp)

# 빈도순으로 정렬한 후 저장
# # Sort decending by frequency
# sorted_filtered_vocab_list = sorted(filtered_tb_vocab_dict.items(), key=lambda x: x[1], reverse=True)
#
# with open('/Users/kintch/Dropbox/sj/2022-2/4. 수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10:3마감/vocab_over10.txt',
#           'w', encoding='utf-8') as f:
#     for keyw, freq in sorted_filtered_vocab_list:
#         f.writelines(str(keyw) + ':' + str(freq) + '\n')

# sorted_filtered_vocab_list[:3]
# [(('다/EF', './SF'), 14653.20606060606), (('ᆫ다/EF', './SF'), 8666.862121212122), (('하/XSV', '어/EC'), 6484.515151515151)]