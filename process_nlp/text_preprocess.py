import re
import spacy
from typing import List
# from utils.config import loggers
# from utils.messages import ErrorMessages, SuccessMessages

# logger = loggers['nlp']

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, do_lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.do_lemmatize = do_lemmatize
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text: str) -> str:
        # logger.info("[TEXT CLEAN] Starting basic cleaning.")
        # Remove non-printable and special characters
        text = re.sub(r"[^\x20-\x7E]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def nlp_process(self, text: str) -> str:
        # logger.info("[TEXT CLEAN] Starting NLP cleaning.")
        doc = self.nlp(text)
        processed_tokens = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if self.remove_stopwords and token.is_stop:
                continue
            if self.do_lemmatize:
                processed_tokens.append(token.lemma_)
            else:
                processed_tokens.append(token.text)
        return " ".join(processed_tokens)

    def preprocess(self, text: str) -> str:
        cleaned = self.clean_text(text)
        final = self.nlp_process(cleaned)
        return final

    def chunk_text(self, text: str, max_chunk_size=1000) -> List[str]:
        # logger.info(f"[CHUNK TEXT] Chunking text of length {len(text)}")
        words = text.split()
        chunks, current_chunk = [], []
        for word in words:
            if sum(len(w) + 1 for w in current_chunk) + len(word) + 1 <= max_chunk_size:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        # logger.info(f"[CHUNK TEXT] Total chunks created: {len(chunks)}")
        return chunks

text_from_pdf = "This is some  raw text, extracted from a PDF!\nIt might have strange spacing... or symbols ."
processor = TextPreprocessor()
preprocessed_text = processor.preprocess(text_from_pdf)
chunks = processor.chunk_text(preprocessed_text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
