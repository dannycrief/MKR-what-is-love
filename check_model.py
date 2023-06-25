import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

# Wczytanie zapisanego modelu
loaded_model = TFBertForSequenceClassification.from_pretrained("drive/MyDrive/000")

# Inicjalizacja tokenizera
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Przykładowy tekst do klasyfikacji
text = "I can't feel anymore"

# Tokenizacja tekstu
encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='tf')

# Klasyfikacja tekstu
outputs = loaded_model(encoded_input)
logits = outputs.logits
predicted_label = tf.argmax(logits, axis=1).numpy()[0]

# Wyświetlenie wyników
if predicted_label == 1:
    print("Tekst wykazuje wyrażenie miłości.")
else:
    print("Tekst nie wykazuje wyrażenia miłości.")
