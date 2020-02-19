import numpy as np
import tensorflow as tf
import pickle
print("[INFO] Loading data")
# Loaded classess
classNames = np.load("model/classes.npy")

# Load tokenizer
with open('model/tokenizer.pickle', 'rb') as handle:
        Tokenizer = pickle.load(handle)


# Load data from file
with open("job_description.txt", "r", encoding="utf8") as file:
    data = file.read()
    jobDescription = [data]


print("[INFO] Loading model")
# Load model
model = tf.keras.models.load_model("model/final_model.hdf5")

# (Don't change)
MAX_LENGTH = 1000

# Preprocess text
import re
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

print("[INFO] Preprocessing")
# Make sequences from given text
jobDescription_processed = Tokenizer.texts_to_sequences(jobDescription)
jobDescription_processed = np.array(jobDescription_processed)
# Pad sequences using post pad
jobDescription_padded = tf.keras.preprocessing.sequence.pad_sequences(jobDescription_processed, padding='post', maxlen=MAX_LENGTH)
# Predict output
print("[INFO] Prediction started")
result = model.predict(jobDescription_padded)

# Get to people
GET_PEOPLE = 5

resultList = list(result[0])

final = []


for i in range(GET_PEOPLE):
    index = resultList.index(max(resultList))
    val = resultList.pop(index)
    final.append([classNames[index], val])


with open("results.txt", "w+") as file:
    for name in final:
        file.writelines("{} - {}\n".format(name[0], round(name[1] * 200, 2)))

print("[INFO] Results saved to 'results.txt'")