import re


# [(형태소1, 품사1), (형태소2, 품사2), ...] 형태로 결과를 리턴


def preprocessor(raw_text):
    '''
    :param raw_text: raw text from (1)pre-processed hwp file or (2) txt file
    :return: list of sentences
    '''
    # 괄호 안과 밖으로 나누어 텍스트 처리
    # 괄호 안 텍스트
    inparen_text = re.findall('\(.*?\)', raw_text)  # non-greedy == .*?
    inparen_text = [x[1:-1] for x in inparen_text]  # 괄호제거
    inparen_sents_list = list(set(sum([re.split(',', x) for x in inparen_text], [])))
    inparen_sents_list = [x.strip() for x in inparen_sents_list]

    # 괄호 밖 텍스트
    outparen_text = re.sub('\(.*?\)', '', raw_text)  # 괄호 안은 삭제
    outparen_text = re.sub('\x07', '', outparen_text)
    outparen_sents_list = re.split('\n', outparen_text)  # 줄바꿈 단위로 나누기
    outparen_sents_list = [x.strip() for x in outparen_sents_list]
    outparen_sents_list = [x for x in outparen_sents_list if x != '']  # 공백 문자 제거

    # 괄호 안 + 괄호 밖
    splited_sents = inparen_sents_list + outparen_sents_list

    return splited_sents
