import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy
import spacy

__all__ = [
    'ttr', 'avg_word_length', 'num_of_char', 'num_of_words', 'bag_of_words',
    'upper_case_ratio', 'pos_ratios', 'first_pers_pron_ratio',
    'comp_superl_ratio', 'citation', 'cardinal', 'tfidf'
]

pos_tags = ["PUNCT", "PRON", "NOUN", "VERB", "ADJ", "ADV"]
adj_comp_ends = ('er', 'erer', 'eres', 'ere', 'eren', 'erem')
adj_superl_ends = ('sten', 'ste', 'ster', 'stes', 'stem')
possPron = [
    "mein", "meine", "meinen", "meiner", "meines", "unser", "unserer",
    "unseres"
]
persPron = ["ich", "mir", "mich", "wir", "uns"]
cardinals = [
    'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun',
    'zehn', 'elf', 'zwölf', 'dreizehn', 'vierzehn', 'fünfzehn', 'sechszehn',
    'siebzehn', 'achtzehn', 'neunzehn'
]
citations = ['\'', '\"']
posRatios = np.zeros(6)

token_char_counts = []


def ttr(doc: spacy.tokens.doc.Doc) -> float:
    list_words_without_punc = [
        token for token in doc.text.split(' ') if token.isalnum()
    ]
    list_unique_words = set(list_words_without_punc)
    count_unique_words = len(list_unique_words)
    no_of_token_without_punc = sum(Counter(list_words_without_punc).values())

    if no_of_token_without_punc != 0:  # preoprocessed data runs into this case
        ttr = count_unique_words / no_of_token_without_punc
    else:
        ttr = 0

    return ttr


def avg_word_length(doc: spacy.tokens.doc.Doc) -> float:
    return len(doc.text) / num_of_words(doc)


def num_of_char(doc: spacy.tokens.doc.Doc) -> float:
    return len(doc.text)


def num_of_words(review_nlp):

    c = Counter(([token.pos_ for token in review_nlp]))
    no_of_tokens = sum(c.values())
    return no_of_tokens


def bag_of_words(text_array: pd.Series) -> scipy.sparse._csr.csr_matrix:
    # inlcude stopwords?
    text_array = text_array.to_list()
    vec = CountVectorizer(ngram_range=(1, 1), stop_words=None).fit(text_array)
    bag_of_words = vec.transform(text_array)

    return bag_of_words


def upper_case_ratio(doc: spacy.tokens.doc.Doc) -> float:
    no_of_chars = len(doc.text)
    return len(re.findall(r"[A-Z]", doc.text)) / no_of_chars


def pos_ratios(review_nlp) -> np.ndarray:
    c = Counter(([token.pos_ for token in review_nlp]))
    no_of_tokens = sum(c.values())
    token_char_counts = []
    for posCat, posCnt in c.items():
        for index, pos_tag in enumerate(pos_tags):
            if posCat == pos_tag:
                posRatios[index] = posCnt / no_of_tokens
    return posRatios


def first_pers_pron_ratio(review_nlp):
    fpp_count = 0
    c = Counter(([token.text for token in review_nlp]))
    no_of_tokens = sum(c.values())
    for posText, posCnt in c.items():
        if posText.lower() in persPron + possPron:
            fpp_count = fpp_count + posCnt

    first_pers_pron_ratio = fpp_count / no_of_tokens
    return first_pers_pron_ratio


def comp_superl_ratio(review_nlp: spacy.tokens.doc.Doc) -> (float, float):
    comp_counter = 0
    superlative_counter = 0
    c = Counter(([token.text for token in review_nlp]))
    no_of_tokens = sum(c.values())
    for index, token in enumerate(review_nlp):
        if len(token.text) > 1:
            token_char_counts.append(len(token.text))
        if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
            if token.text.endswith(adj_comp_ends):
                comp_counter += 1
            if token.text.endswith(adj_superl_ends):
                superlative_counter += 1

    comp_ratio = comp_counter / no_of_tokens
    superl_ratio = superlative_counter / no_of_tokens
    return comp_ratio, superl_ratio


def citation(text) -> int:

    list_citation_token = [token for token in text if str(token) in citations]

    count_citation = len(list_citation_token)
    return count_citation


def cardinal(doc: spacy.tokens.doc.Doc) -> int:
    list_cardinal = [
        token for token in doc.text.split(' ') if token in cardinals
    ]
    count_cardinal = len(list_cardinal)
    return count_cardinal


def tfidf(text):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    return X


def sentiment(text):
    pass


doc = """Trotz aller Bedenken scheint das Abkommen, das die Türkei und die UN mit Russland und der Ukraine geschlossen hat, zu halten: Das Schwarze Meer ist wieder schiffbar.
Das ist ein weiterer Baustein dazu, die großen Befürchtungen aus der ersten Phase des Krieges abzubauen. Beobachter fürchteten damals eine Hungerwelle über den Globus, falls die Ukraine als Exporteur komplett ausfallen sollte. Doch dazu ist es nicht gekommen. Dafür gibt es mehrere Gründe."""


def main():
    print('Testing feature: TTR', ttr(doc))


if __name__ == "__main__":
    main()
