from scripts.preprocess import Preprocess
from scripts.utils import Utils
from nltk import word_tokenize
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

preprocess = Preprocess()
utils = Utils()




DATA_DIR = "../data"


print("[INFO] Starting preprocessing")
df = preprocess.getDataFrame(DATA_DIR)

y = df["label"].to_numpy()
X = df["text"].to_numpy()

print("Training data shape:", X.shape)
print("Training labels shape:", y.shape)


print("\n[INFO] Tokenizing data")
tokens = [word_tokenize(sen) for sen in X]

# Lower tokens
lower_tokens = [utils.lower_token(token) for token in tokens]

# Remove stopwords
filtered_words = [utils.removeStopWords(sen) for sen in lower_tokens]
text_final = [' '.join(sen) for sen in filtered_words]
tokens_final = filtered_words

# Basic variables
vocab_size = 10000
embedding_dim = 16
max_length = 1000
trunc_type = "post"
oov_tok = "<OOV>"

# Tokenize words
tokenizer = Tokenizer(num_words=vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(text_final)
word_index = tokenizer.word_index

# Pad sequences
sequences = tokenizer.texts_to_sequences(text_final)
padded = pad_sequences(sequences, maxlen = max_length, truncating=trunc_type, padding='post')

print("\nINFO: Loading Embedding")
# Load embeddings
embeddings_dictionary = dict()
glove_file = open("glove.6B.100d.txt", encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector



encoder = LabelBinarizer()
y = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.3)
print("Training data:", X_train.shape)
print("Test data:", X_test.shape)


print("\nINFO: Starting training")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(91, activation='softmax')
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
num_epochs = 15
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data = (X_test, y_test))

print("\nINFO: Saving Model")

model.save("model/final_model.hdf5")
# Save classes file
np.save("model/classes.npy", encoder.classes_)

# Save tokens file
import pickle
with open('model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)