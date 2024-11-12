import spacy
import string
from bs4 import BeautifulSoup

def preprocess_text(text):
    """
    Preprocesses text by removing HTML tags, lowercasing, handling punctuation,
    and lemmatizing using spaCy.

    Args:
        text: The input text string.

    Returns:
        The preprocessed text string.
    """
    # 1. Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    # 2. Initialize spaCy
    nlp = spacy.load("en_core_web_sm")  # Make sure you've installed en_core_web_sm

    # 3. Process the text with spaCy
    doc = nlp(text)

    # 4. Preprocessing steps
    processed_tokens = []
    for token in doc:
        # a. Lowercasing (already handled by spaCy tokenizer)
        # b. Punctuation removal (optional, customize as needed)
        if token.text not in string.punctuation: # basic punctuation removal
           # further processing if not punctuation
           if not token.is_stop: #optional stop word removal
               # Lemmatization
               processed_tokens.append(token.lemma_.lower())

    # 5. Join processed tokens back into a string
    preprocessed_text = " ".join(processed_tokens)
    return preprocessed_text


# Example usage:
text = "<p>The COVID-19 pandemic has <em>affected</em> economies worldwide!</p>"
preprocessed_text = preprocess_text(text)
print(preprocessed_text)  # Output: covid-19 pandemic affect economy worldwide