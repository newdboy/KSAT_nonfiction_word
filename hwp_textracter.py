import re
import olefile
import zlib
import struct

# 확장자 .hwp 파일의 텍스트를 추출
# input: filename > output: post-processed text
## 이후 *.doc 형태도 제공하도록 보강 필요

def get_hwp_text(filepath):
    with olefile.OleFileIO(filepath) as f:
        dirs = f.listdir()

        # HWP 파일 검증
        if ["FileHeader"] not in dirs or \
                ["\x05HwpSummaryInformation"] not in dirs:
            raise Exception("Not Valid HWP.")

        # 문서 포맷 압축 여부 확인
        header = f.openstream("FileHeader")
        header_data = header.read()
        is_compressed = (header_data[36] & 1) == 1

        # Body Sections 불러오기
        nums = []
        for d in dirs:
            if d[0] == "BodyText":
                nums.append(int(d[1][len("Section"):]))
        sections = ["BodyText/Section" + str(x) for x in sorted(nums)]

        # 전체 text 추출
        text = ""
        for section in sections:
            bodytext = f.openstream(section)
            data = bodytext.read()
            if is_compressed:
                unpacked_data = zlib.decompress(data, -15)
            else:
                unpacked_data = data

            # 각 Section 내 text 추출
            section_text = ""
            i = 0
            size = len(unpacked_data)
            while i < size:
                header = struct.unpack_from("<I", unpacked_data, i)[0]
                rec_type = header & 0x3ff
                rec_len = (header >> 20) & 0xfff

                if rec_type in [67]:
                    rec_data = unpacked_data[i + 4:i + 4 + rec_len]
                    section_text += rec_data.decode('utf-16', errors='ignore')
                    section_text += "\n"

                i += 4 + rec_len

            text += section_text
            text += "\n"

        # post-processing noise included text
        noise_included = text[16:]  # [0:16]은 쓸모 없는 내용
        pp_text = re.sub("[^가-힣A-Za-z0-9 ]{1}x{1}[^가-힣]*", "", noise_included)  # \x00 과 같은 단어들 일괄 제거
        pped_text = re.sub("[^가-힣A-Za-z0-9 -=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\n]", "", pp_text)  # 한글, 영어, 특수문자 제외 하구 모두 제거

        # good by nogada (be going to delete code below ...)
        # pped_text = re.sub("[一-龥]", "", pp_text)
        # pp_text = re.sub("\x0b漠杳\x00\x00\x00\x00\x0bs", "", pp_text)
        # pp_text = re.sub("\x15湯湷\x00\x00\x00\x00\x15", "", pp_text)
        # pp_text = re.sub("\x15湰灧\x00\x00\x00\x00\x15", "", pp_text)
        # pp_text = re.sub("\x0b漠杳\x00\x00\x00\x00\x0b", "", pp_text)
        # pp_text = re.sub("\x10慤桥\x00\x00\x00\x00\x10", "", pp_text)
        # pp_text = re.sub("\x10慤桥\x00\x00\x00\x00\x10", "", pp_text)
        # pp_text = re.sub("\x15桤灧\x00\x00\x00\x00\x15", "", pp_text)
        # pped_text = re.sub("\x12湯慴\x00\x00\x00\x00\x12", "", pp_text)

        return pped_text

