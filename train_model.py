import pandas as pd
import tensorflow as tf
from transformers import InputExample, BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import glue_convert_examples_to_features

# Wczytanie danych
data = pd.read_csv('csv_files/reddit_posts_20230503.tsv', sep='\t', lineterminator='\n')

# Przygotowanie etykiet
data['label'] = data['selftext'].apply(lambda text: 1 if "love" in text.lower() else 0)

# Podział na zbiór treningowy i testowy
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Inicjalizacja tokenizera i modelu
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')


# Funkcja convert_examples_to_tf_dataset konwertuje InputExamples na odpowiedni format danych do trenowania modelu
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = glue_convert_examples_to_features(examples=examples, tokenizer=tokenizer, max_length=max_length,
                                                 task='sst-2', label_list=['0', '1'])

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(gen, (
        {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64), (
                                              {"input_ids": tf.TensorShape([None]),
                                               "attention_mask": tf.TensorShape([None]),
                                               "token_type_ids": tf.TensorShape([None])}, tf.TensorShape([])))


# Konwersja InputExamples na TensorFlow Dataset
train_examples = train.apply(lambda x: InputExample(guid=None,
                                                    text_a=x['selftext'],
                                                    text_b=None,
                                                    label=str(x['label'])),
                             axis=1)

train_data = convert_examples_to_tf_dataset(list(train_examples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

test_examples = test.apply(lambda x: InputExample(guid=None,
                                                  text_a=x['selftext'],
                                                  text_b=None,
                                                  label=str(x['label'])),
                           axis=1)

test_data = convert_examples_to_tf_dataset(list(test_examples), tokenizer)
test_data = test_data.batch(32)

# Kompilacja i trening modelu
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=2, validation_data=test_data)

# Zapisanie modelu
model.save_pretrained("models")
