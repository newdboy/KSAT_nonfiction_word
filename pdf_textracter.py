from PyPDF2 import PdfReader

def pdf_to_text(filepath):
    def preproc_on_pdftotxt(txt):
        txt = txt.encode().decode().replace("\x00", " ")
        # txt = txt.replace("\n", "")  # exsist in text_preprocessor
        return txt

    reader = PdfReader(filepath)
    pages = reader.pages
    text = ""
    for page in pages:
        sub = page.extract_text()
        text += sub
    text = preproc_on_pdftotxt(text)
    return text
