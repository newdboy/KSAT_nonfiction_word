from vocab_tools import vocab_extractor

'''
main.py: EXTRACT WORD/N-GRAM WORDS FROM INPUT TEXTS

it includes functions below.

Function 0. IMPORT files
- Import all files inside a folder.
(This function takes into account the subfolder structures.)

Function 1. EXTRACT texts
- Extract texts inside the files.
(files must be *.pdf or *.hwp)

Function 2. EXTRACT words from texts
- If the word you are extracting is a word with one morpheme, use the part-of-speech tag filter; 
if the word is a word with more than one morpheme, extract all n-gram words that appear more than once.
- Don't do frequency-based filtering in this session because you'll need to consider the result of multiple files combined together;
do it later when you combine the files.

Function 3. SAVE results
- Save to pickles
'''

## DEFINE VARIABLES
# Part-of-speech templates to base lexical extractions on
# [('EC',)
templates = [
    # 한 단어로 구성된
    ('NNG',),
    ('NNP',),
    ('VV',),
    ('VA',),
    ('VV-I',),
    ('VA-I',),
    ('VV-R',),
    ('VA-R',),
    ('MAG',),
    ('XR',),
    ('SL',),

    # 두 단어로 구성된
    ('NNG', 'NNG',),
    # ('NNG', 'SO',),
    ('NNG', 'XSA',),
    ('NNG', 'XSA-I',),
    ('NNG', 'XSA-R',),
    ('NNG', 'XSN',),
    ('NNG', 'XSV',),
    ('NP', 'NNG',),
    ('SN', 'NNG',),
    ('SN', 'SL',),
    ('SL', 'NNG',), #DNA폴리머라아제, QR코드
    ('XR', 'XSN',), #어근+접미사
    ('XR', 'XSV',),
    ('XR', 'XSA',),
    ('XR', 'XSA-I',),
    ('XR', 'XSA-R',),
    ('XR', 'XSM',),
    ('XPN', 'NNG',),#접두사+명,동,형
    ('XPN', 'VV',),
    ('XPN', 'VA',),
    ('XPN', 'VV-I',),
    ('XPN', 'VA-I',),
    ('XPN', 'VV-R',),
    ('XPN', 'VA-R',),



    # 세 단어로 구성된
    ('SN', '', 'NNG',),  #3D프린터, 2가염색체
    ('SL', '', 'NNG',),  #DNA중합효소
    ('SN', '', '', 'NNG',),  #3D레이져프린터
    ('NNG', 'JC', 'NNG',),
    ('NNG', 'JC', 'NNP',),
    ('NNG', 'JKG', 'NNG',),
    ('NNG', 'JKO', 'NNG',),
    ('NNG', 'JKO', 'VV-I',),
    ('NNG', 'JKO', 'VV-R',),
    ('NNG', 'JKO', 'VV',),
    ('NNG', 'NNG', 'NNG',),
    ('NNG', 'NNG', 'XSN',),

    # 네 단어로 구성된
    ('NNG', 'NNG', 'NNG', 'NNG',),
    ('SN', 'NNG', 'NNG', 'NNG',),  #('3SN', '차원NNG', '공간NNG', '좌표NNG'),
    ('SN', 'SL', 'NNG', 'NNG',),  #('3SN', 'DSL', '레이저NNG', '스캐너NNG'),
    ('SN', 'NNB', 'NNG', 'NNG',)  #('2SN', '차NNB', '세계NNG', '대전NNG'),
]
    # ('/NNG',), ('/NNP',), ('/VV',), ('/VA',), ('/MAG',), ('/XR',), ('/SL',),  # mono-gram
    # ('',) * 2,
    # ('',) * 3,
    # ('',) * 4

tdir = {
    'ksat': '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/텍스트 자료/수능 비문학 텍스트 자료/수능txt',
    'ebs': '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/텍스트 자료/ebs 텍스트 자료',
    'textbook': '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/텍스트 자료/교과서 텍스트 자료'
}
pickle_save_dir_ground0 = "/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/5. pickles/"
ngram_range = (1, 4)  # (a,b); from a to b (not b-1)


## EXTRACT VOCABULARY (for each folders & files) & SAVE TO PICKLES
vocab_extractor(source_text_dir=tdir, pos_templates=templates, pickle_save_dir=pickle_save_dir_ground0, min_count_num=1, ngram_range=(1,4))


# test용 저장
# with open('/Users/kintch/Dropbox/sj/2022-2/4. 수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10:3마감/텍스트 자료/test.txt',
#           'w', encoding='utf-8') as f:
#     for ngram, freq in counted_ngram_5.items():
#         f.writelines(str(ngram) + ':' + str(freq) + '\n')

# 어휘 추출

# outparallel_2_spaced = [spacing(x) for x in outparallel_2]

# from tqdm import tqdm
# outparallel_2_spaced = list()
# for x in tqdm(outparallel_2):
#     outparallel_2_spaced.append(spacing(x))


# test ----------------------------------------------------
# test= '분열 감염성 질병 갑상샘 개체 개체 수 개체군 개체군의 밀도 개체군의 생장 겉질 결실 고사량 고양이 울음 증후군 고유종 고지혈증 골격근 공생 과분극 관목 교감 신경 구균 구심성 뉴런 군집 군체 귀납적 탐구 방법'
# test1 = '2 가 염색체 2 차 면역 반응 2 차 소비자 2 차 천이 II형 생존 곡선 이화 작용 인슐린 인플루엔자 바이러스 1 차 면역 반응 1 차 소비자 1 차 천이 I형 생존 곡선'
# test2 = '생물다양성이감소하는원인은외래종의도입,서식지파괴와 단편화,불법포획과남획,환경오염및기후변화등이있다.'
# test3 = '버들붕어 개체군 내에서는 텃세가 나타난다. 자신의 세 력권을 형성하여 자신의 영역을 유지하며 다른 개체 와의 경 쟁을 줄여 개체군을 유지한다.'


# a = '형질 내세망'
# b = '인슐린 의존성 당뇨'
# c = 'ⓐ와 ⓑ는 ATP와 ADP +Pi 중 하나이다.'


# from mecab import MeCab
# mecab = MeCab()
#
# mecab.pos('생물다양성은철도,도로등의건설로인한단편화,외래종의 무차별적인도입,무분별한남획등에의해감소한다.')
# spacing()
# okt.pos('생물다양성은철도,도로등의건설로인한단편화,외래종의 무차별적인도입,무분별한남획등에의해감소한다.', norm=True)
