from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()

    # Tokenize the cleaned text
    tokens = word_tokenize(clean_text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords and apply lemmatization
    stop_words = set(stopwords.words('english'))
    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            lemmatized_token = lemmatizer.lemmatize(token)  # Apply lemmatization
            processed_tokens.append(lemmatized_token)

    return " ".join(processed_tokens)

html_string = "<p>This is an <b>example</b> string.</p>"
clean_string = preprocess_text(html_string)
print(clean_string)  # Output: this example string