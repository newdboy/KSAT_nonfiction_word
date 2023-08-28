import fitz
dir_path = '/Users/kintch/Library/CloudStorage/Dropbox/Mac/Downloads/2023학년도 경서중학교 1학기 전과목 평가계획 pages 1 - 42_ (1).pdf'

tm = [
    # ('/NNG', '/XSN',),
    # ('/NNG', '', '/XSN',),
    ('/NNG', '/NNG', '/NNG', '/XSN',),
    # ('/SL', '/NNG'), #DNA폴리머라아제, QR코드
    # ('/SN', '', '/NNG'),  #3D프린터, 2가염색체
    # ('/SL', '', '/NNG'), #DNA중합효소
    # ('/SN', '', '', '/NNG')  #3D레이져프린터
# ('/SL', '', '', '/NNG')  #3D레이져프린터
]
from pprint import pprint
from vocab_tools import templated_ngram_extractor
from hwp_textracter import get_hwp_text
from text_preprocessor import preprocessor
rt = get_hwp_text('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/텍스트 자료/ebs 텍스트 자료/2022/EBS 2022학년도 수능특강 국어영역 독서.hwp')

fp = '/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-1/연구/[진행중]수능 비문학지문 친숙하지 않은 어휘 (독서학회) 10_3마감/텍스트 자료/수능 비문학 텍스트 자료/수능txt/2018/2018.txt'
with open(fp, encoding='utf-8') as f:
    raw_text = f.read()

sents = preprocessor(rt)
res = templated_ngram_extractor(docs=sents, pos_templates=tm)
pprint(res)


for a in tqdm.tqdm(sents):
    # print(a)
    if 'DNA' in a:
        print('yes')
        print(a)

pprint(res)
from kiwipiepy import Kiwi
kiwi = Kiwi(num_workers=4, model_type='sbg', typos='basic')
kiwi.cutoff_threshold = 5
kiwi.tokenize('민주주의 역할론')
kiwi.tokenize('DNA중합효소')
kiwi.tokenize('DNA폴리머라아제')