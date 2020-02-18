from nltk.corpus import stopwords

class Utils():
    def __init__(self):
        pass

    def removeStopWords(self, tokens): 
        stoplist = stopwords.words('english')
        return [word for word in tokens if word not in stoplist]

    def lower_token(self, tokens): 
        return [w.lower() for w in tokens]