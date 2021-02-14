import nltk
import re

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import unicodedata

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

nltk.download("maxent_ne_chunker")
nltk.download("words")

import spacy

DOMAIN_STOPWORDS = [
    "lady",
    "sir",
    "gentleman",
    "madam",
    "sirs",
    "madams",
    "dear",
    "gentlemen",
    "ladies",
    "hello",
    "zalando",
    "team",
    "zalandoteam",
    "sincerely",
    "regards",
    "lg",
    "mfg",
    "vg",
    "hi",
    "hey",
    "greeting",
]
NGRAM_STOPWORDS = [
    "best regards",
    "yours sincerely",
    "many thanks",
    "good bye",
    "good day",
    "good morning",
    "good evening",
    "kind regards",
    "thank you",
    "happy easter",
    "merry christmas",
    "thanks in advance",
    "friendly greeting",
    "nice day",
]

stop_words_list = stopwords.words("english")
STOPWORDS = stop_words_list + DOMAIN_STOPWORDS

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def remove_accented_chars(text):
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text, remove_digits=True):
    pattern = r"[^a-zA-z0-9\s]" if not remove_digits else r"[^a-zA-z\s]"
    text = re.sub(pattern, " ", text)

    # Remove words containing at least 3 'x'
    text = re.sub(r"(\b\w*[x,X]{3,}\w*\b)", "", text)
    text = " ".join(text.split())

    return text


def remove_ngram_stopwords(text, ngram_stopwords=NGRAM_STOPWORDS):
    texts_lower = text.lower()
    reg = re.compile("|".join(map(re.escape, ngram_stopwords)))
    text_filtered = reg.sub("", texts_lower)
    text = " ".join(text_filtered.split())
    return text


def remove_stopwords(text, stopwords_list=STOPWORDS):
    texts_lower = text.lower()
    word_list = texts_lower.split()
    text_filtered_list = [word for word in word_list if word not in stopwords_list]
    text_filtered = " ".join(text_filtered_list)
    return text_filtered


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()

    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith("J"):
            return wordnet.ADJ
        elif nltk_tag.startswith("V"):
            return wordnet.VERB
        elif nltk_tag.startswith("N"):
            return wordnet.NOUN
        elif nltk_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_text = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_text.append(word)
        else:
            lemmatized_text.append(lemmatizer.lemmatize(word, tag))

    return " ".join(lemmatized_text)


def normalize_text_list(texts):
    normalized_text = []
    for text in texts:
        text_tmp = remove_html_tags(text)
        text_tmp = remove_accented_chars(text)
        text_tmp = expand_contractions(text_tmp)
        text_tmp = remove_special_characters(text_tmp)
        text_tmp = remove_ngram_stopwords(text_tmp)
        text_tmp = remove_stopwords(text_tmp)
        text_tmp = lemmatize_text(text_tmp)
        normalized_text.append(text_tmp)

    return normalized_text


def remove_names(texts):
    texts_cleaned = []
    for text in texts:
        person_list = []
        person_names = person_list
        tokens = nltk.tokenize.word_tokenize(text)
        pos = nltk.pos_tag(tokens)
        sentt = nltk.ne_chunk(pos, binary=False)

        person = []
        name = ""
        for subtree in sentt.subtrees(filter=lambda t: t.label() == "PERSON"):
            for leaf in subtree.leaves():
                person.append(leaf[0])
            if len(person) > 1:
                for part in person:
                    name += part + " "
                text.replace(name, "xxx")
                if name[:-1] not in person_list:
                    person_list.append(name[:-1])
                name = ""
            person = []

        texts_cleaned.append(text)

        for person in person_list:
            person_split = person.split(" ")
            for name in person_split:
                if wordnet.synsets(name):
                    if name in person:
                        person_names.remove(person)
                        break

        print(f"person_names: {person_names}")
    return texts_cleaned


def remove_names_spacy(texts):
    nlp = spacy.load("en_core_web_sm")
    texts_cleaned = []
    for text in texts:
        doc = nlp(text)
        persons = [i for i in doc.ents if i.label_.lower() in ["person"]]
        if len(persons):
            for p in persons:
                text = text.replace(str(p), "xxx")

        texts_cleaned.append(text)
    return texts_cleaned
