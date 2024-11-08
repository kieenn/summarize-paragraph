import heapq
import re
import unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

def preprocess_text(text):
    text = text.lower()
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()

    # Tokenize the cleaned text
    tokens = word_tokenize(clean_text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Apply lemmatization without removing stopwords
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(processed_tokens)

def summarize_text(text):
    """
    Summarizes a given text using TF-IDF and sentence ranking.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """

    # 1. Preprocess Text
    processed_text = preprocess_text(text)
    # print(processed_text)
    # Check if processed_text is empty to avoid ValueError
    if not processed_text.strip():
        return "No valid content to summarize."

    # 2. Create TF-IDF Vectorizer
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([processed_text])

    # 3. Calculate Sentence Scores
    sentence_scores = tfidf_matrix.toarray().sum(axis=1)
    ranked_sentences = sorted(zip(sentence_scores, sent_tokenize(processed_text)), reverse=True)

    # 4. Generate Summary (Include all sentences)
    summary = " ".join([sentence for _, sentence in ranked_sentences])
    return summary

def summarize_text2(text):
    text = preprocess_text(text)
    """
    Summarizes a given text using LexRank summarization,
    automatically determining the number of sentences.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    # Calculate the number of sentences based on the text's length
    sentence_count = max(2, int(len(sent_tokenize(text)) * 0.2))

    summary = summarizer(parser.document, sentence_count)
    return " ".join([sentence._text for sentence in summary])


import nltk

def summarize_text3(text):
    """
    Summarizes the given text using a frequency-based approach.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: A summary of the input text.
    """
    #text = preprocess_text(text)
    # Tokenize sentences and words
    sentence_list = nltk.sent_tokenize(text)
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in nltk.corpus.stopwords.words('english'):
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Calculate word frequencies relative to the maximum frequency
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    # Score sentences based on word frequencies
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # Extract top 7 sentences based on scores
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    # Join sentences into a summary
    summary = ' '.join(summary_sentences)

    return summary


def text_similarity(text1, text2):
    # Tokenize and lemmatize the texts
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in word_tokenize(text1)]
    tokens2 = [lemmatizer.lemmatize(token) for token in word_tokenize(text2)]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform([' '.join(tokens1)])
    vector2 = vectorizer.transform([' '.join(tokens2)])

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)[0][0]

    return similarity

def calculate_text_similarity(text1, text2):
    """
    Calculates the cosine similarity between two texts using TF-IDF.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The cosine similarity score (between 0 and 1).
    """

    # Convert the texts into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity between the vectors
    similarity = cosine_similarity(vectors)[0, 1]  # Get the similarity between text1 and text2

    return similarity
