from kiwipiepy import Kiwi
from collections import defaultdict
from text_preprocessor import preprocessor
import os
from hwp_textracter import get_hwp_text
from pdf_textracter import pdf_to_text
import pickle
from dict_tools import dict_merger
import re


# Source code CITATION: https://lovit.github.io/nlp/2018/10/23/ngram/
# Attribute LOVIT's idea

# for x, y in zip([1,2,3], [4,5,6]):
#     print(x,y)

# 추출한 n-gram 중에서 특정 pos-template을 지니는 n-gram만을 다시 추출한다.
def matcher(ngram, template):
    # re.match와 혼동 가능하기 때문에 matcher로 변경
    if len(ngram) != len(template):
        # print('not same length with', ngram, r'/', template)
        return False
    for n, t in zip(ngram, template):
        if not (t == n[1]):
            return False
            break
        else:
            continue
    return True

# test_ng = (('이기', 'EC'), ('어', 'EC'))
# test_tem = ('EC','EC',)
# len(test_tem)
# matcher(test_ng, test_tem)
# --delete below
# test_ng = ('ᆯ/ETM', '계기/NNG')
# test_tem = ('/E',)


# define tokenizer using kiwi
# 형태소 분석 결과 추가 정의 (add_pre_analyzed_word)
import pickle
# with open('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/noun_word_list.pkl','rb') as f:
#     noun_word_list = pickle.load(f)

kiwi = Kiwi(num_workers=4, model_type='sbg')  # , typos='basic
kiwi.add_pre_analyzed_word('실세계', [('실','NNG'), ('세계', 'NNG')], 100)
kiwi.add_pre_analyzed_word('격자판', [('격자', 'NNG'), ('판','NNG')], 100)
kiwi.add_pre_analyzed_word('신용도', [('신용', 'NNG'), ('도','NNG')], 100)
kiwi.add_user_word('내인성', 'NNP', 100)
# for x in noun_word_list:
#     kiwi.add_user_word(x, 'NNP', 100)




def tokenizer_kiwi(sent, pre_spacing=False):  # [(형태소1, 품사1), (형태소2, 품사2), ...] 형태로 결과를 리턴
    if pre_spacing:
        sent = kiwi.space(sent, reset_whitespace=True)
    result = list(map(lambda x: (x[0], x[1]), kiwi.tokenize(sent)))
    return result

# kiwi.tokenize('http://')
# tokenizer_kiwi('http://')
# test = tokenizer_kiwi('http://')
# [x.split('/') for x in test]  # 이렇게 하면 안되겠다. string 안에 /가 있는 경우 다 나눠버림. (예:http://)

# 빈도 / POS template 로 filtering 가능한 n-gram counter
# neg_templates = [('',) * 999]
# pos_templates = [('',) * k for k in (1,5)]

# tokenizer_kiwi('코로나를 이겨냅시다 으쌰으쌰')

# templated_ngram_extractor(docs, pos_templates)


def templated_ngram_extractor(docs, pos_templates=list(), neg_templates=list(),
                              min_count=5, n_range=(1, 5)):
    '''
    :param docs: list of doc (e.g.) [doc1, doc2, ...]
    :param pos_templates: list of templates (e.g.) [tuple(['/NN']), ...] or [(/NN,), ...]
    :param neg_templates:
    :param min_count:
    :param n_range:
    :return:
    '''


    def make_ngrams(adjacent_words: list, n: int):
        ngrams = []
        for b in range(0, len(adjacent_words) - n + 1):
            ngrams.append(tuple(adjacent_words[b:b + n]))
        return ngrams


    def filter_ngram_from_templates(ngrams, pos_temps=list(), neg_temps=list()):
        matcheds = []
        for ngram in ngrams:

            if pos_temps:
                for pos_template in pos_temps:  # pos_temps 이 있는 경우
                    # print("pos_template: ", pos_template)
                    if matcher(ngram, pos_template):

                        if neg_temps:  # neg_temps 이 있는 경우
                            ox_tester = False  # 기본값 False
                            for neg_template in neg_temps:
                                if matcher(ngram, neg_template):
                                    ox_tester = False
                                    break
                                else:
                                    ox_tester = True
                                    continue  # 수정
                            if ox_tester:
                                matcheds.append(ngram)  # 모두 잘 통과하면 추가
                        else:
                            matcheds.append(ngram)

            else:
                if neg_temps:
                    ox_tester = False  # 기본값 False
                    for neg_template in neg_temps:
                        if matcher(ngram, neg_template):
                            ox_tester = False
                            break
                        else:
                            ox_tester = True
                    if ox_tester:
                        matcheds.append(ngram)

                else:
                    matcheds.append(ngram)

        return matcheds

    #test
    # docs = ['http://', '코로나를 이겨냅시다 으쌰으쌰 으쌰 진성 아구찜']
    # pos_templates = [('VV', '', ''), ('IC', '')]
    # neg_templates = list()
    # min_count = 1
    # n_range = (1, 5)

    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in docs:
        tokenized_words_list = tokenizer_kiwi(doc)
        # print(tokenized_words_list)
        for n in range(n_begin, n_end + 1):
            ngrams = make_ngrams(tokenized_words_list, n)
            # print(ngrams)
            ngrams = filter_ngram_from_templates(ngrams=ngrams, pos_temps=pos_templates, neg_temps=neg_templates)
            for ngram in ngrams:
                ngram_counter[ngram] += 1

    ngram_counter = {
        ngram: count for ngram, count in ngram_counter.items()
        if count >= min_count
    }
    from pprint import pprint
    # pprint(ngram_counter)

    return ngram_counter



    # (for test)
    # source_text_dir = tdir
    # pos_templates = [('SN', 'SL',)]
    # pickle_save_dir = pickle_save_dir_ground0
    # min_count_num = 1
    # ngram_range = (1, 4)

def vocab_extractor(source_text_dir, pos_templates, pickle_save_dir, min_count_num=1, ngram_range=(1, 4)):
    for text_type, path in source_text_dir.items():
        pickle_save_dir_ = pickle_save_dir + text_type + '/'
        folder_path = path

        # Make file path (folder path & file name) list
        file_path_list = list()
        for (root, directories, files) in os.walk(folder_path):
            file_path_list.append([root, files])  # [루트, [파일명1, 파일명2 ...]] (list)을 추출
        # len(file_path_list)  # 파일 몇 개?

        n = 0
        while n < len(file_path_list):
            folder_path = file_path_list[n][0]
            folder_name = re.sub(r'/.*/', '', folder_path)
            print('===', 'FOLDER NAME: ', folder_name, '===')
            k = 0
            vocab_dict = dict()
            if ".DS_Store" in file_path_list[n][1]:
                file_path_list[n][1].remove(".DS_Store")
            while k < len(file_path_list[n][1]):
                file_name = file_path_list[n][1][k]
                if file_name != ".DS_Store":
                    file_path = os.path.join(folder_path, file_name)
                    print('(', k + 1, r'/', len(file_path_list[n][1]), ')' 'WORKING FILE NAME:', file_name, )
                    if '.txt' in file_name:
                        with open(file_path, encoding='utf-8') as f:
                            raw_text = f.read()

                    elif '.hwp' in file_name:
                        raw_text = get_hwp_text(file_path)

                    elif '.pdf' in file_name:
                        raw_text = pdf_to_text(file_path)

                    # 전처리
                    sents = preprocessor(raw_text)

                    # 토큰화 & 템플릿 기반 n-gram 추출 (monogram 포함)
                    counted_ngram_5 = templated_ngram_extractor(sents,
                                                                pos_templates=pos_templates,
                                                                min_count=min_count_num,
                                                                n_range=ngram_range)
                    # print(counted_ngram_5)
                    # vocab_dict에 추가
                    vocab_dict = dict_merger(vocab_dict, counted_ngram_5)
                k += 1

            # Save vocabulary with pickle
            if vocab_dict:
                if not os.path.exists(pickle_save_dir_):
                    try:
                        os.mkdir(pickle_save_dir_)
                    except:
                        pass
                pickle_file_dir = pickle_save_dir_ + folder_name + '_' + text_type + '_vocab.pickle'  # ignore files' name, reflect each folders name to its file name.
                with open(pickle_file_dir, 'wb') as wp:
                    pickle.dump(vocab_dict, wp)
            n += 1


def templated_ngram_dict_filter(ngram_dict, pos_templates=list(), neg_templates=list()):
    """
    :param ngram_dict: dict of ngram_word / freq
    :param pos_templates: list of templates
        (e.g.) [tuple(['/NN']), tuple(['/NNG', '/NNB']) ]
                or [(/NN,), (/NNB, /NNG)]
    :param neg_templates:
    :return:
    """

    matched_dict = dict()
    for ngram, freq in ngram_dict.items():

        if pos_templates:
            for pos_template in pos_templates:  # pos_templates 이 있는 경우
                if matcher(ngram, pos_template):

                    if neg_templates:
                        ox_tester = False
                        for neg_template in neg_templates:
                            if matcher(ngram, neg_template):
                                ox_tester = False
                                break  # 더 해볼 것도 없다.
                            else:
                                ox_tester = True
                                continue  # 일치하면 볼 것도 없이 종료
                        if ox_tester:  # -1은 리스트 요소가 하나일 경우 대비.
                            matched_dict[ngram] = freq  # 끝까지 살아 남으면 추가
                    else:  # neg_templates 이 없는 경우
                        matched_dict[ngram] = freq  ## positive는 하나라도 걸리면 추가
                else:
                    pass

        else:
            if neg_templates:  # neg_templates 이 있는 경우
                ox_tester = False
                for neg_template in neg_templates:  # neg_templates에서 하나 뽑기

                    if matcher(ngram, neg_template):
                        ox_tester = False
                        break  # 일치하면 다른 거 더 해볼 일도 없음.
                    else:
                        ox_tester = True
                        # 네거티브 탬플릿과 일치하지 않는거면 일단 진행.
                        # 다음, 그 다음 네거티브 탬플릿과도 일치하지 않아야 함
                        continue
                if ox_tester:  # 최종 판단
                    matched_dict[ngram] = freq  # 끝까지 살아 남으면 추가
            else:  # neg_templates 이 없는 경우
                matched_dict[ngram] = freq  # 바로 추가

    return matched_dict

# test = {('으로/JKB', '독일/NNP'): 1, ('실세/NNG', '계/XSN', '를/JKO'): 1, ('ᆯ/ETM', '계기/NNG'): 1,
#         ('도/JX', '붕괴/NNG', '되/XSV', 'ᆯ/ETM'): 1}
#
# templated_ngram_dict_filter(test, neg_templates=[('/', '/')])
