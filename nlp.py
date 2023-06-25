import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Załaduj zestaw danych
df = pd.read_csv('csv_files/reddit_posts_20230503.tsv', sep='\t')

# Pobierz zasób 'punkt' z NLTK, jeśli jeszcze go nie masz
nltk.download('punkt')

# Pobierz zasób 'stopwords' z NLTK, jeśli jeszcze go nie masz
nltk.download('stopwords')

# Utwórz listę słów kluczowych związanych z miłością
love_keywords = ['love', 'affection', 'adoration', 'amour', 'devotion', 'passion', 'adoration']


# Zdefiniuj funkcję do identyfikacji wyrażeń miłości
def identify_love_expressions(text: str):
    # Tokenizacja słów
    words = word_tokenize(text)

    # Usunięcie stopwords
    words = [word for word in words if word not in stopwords.words('english')]

    # Zliczanie wystąpień wyrażeń miłości
    love_counts = Counter(word for word in words if word in love_keywords)

    return love_counts


# Zastosuj funkcję do identyfikacji wyrażeń miłości na ramce danych
df['love_counts'] = df['selftext'].apply(identify_love_expressions)

# Wyświetl wyniki
print(df['love_counts'])
