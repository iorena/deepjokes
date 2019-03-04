
class DataPrepper:
    def __init__(self, data, cutoff):
        self.index = 0
        self.allData = self.cleanup(data)
        self.createBOW(cutoff)
        self.prepareData()

    def createBOW(self, cutoff):
        words = {}
        for joke in self.allData:
            self.countwords(words, joke["openingLine"])
            self.countwords(words, joke["punchline"])
        self.contentBOW = {word: words[word] for word in words if words[word]["count"] < cutoff}
        self.fillerBOW = {word: words[word] for word in words if words[word]["count"] >= cutoff}

    def prepareData(self):
        self.data = {}
            print(len(self.allData))
        self.data["train"] = self.allData[1:20000]
        self.data["test"] = self.allData[20001:]

    def countwords(self, words, line):
        for word in line:
            if word in words:
                words[word]["count"] = words[word]["count"] + 1
            else:
                words[word] = {}
                words[word]["count"] = 1
                words[word]["id"] = self.index
                self.index += 1

    def cleanup(self, data):
        cleanedData = []
        for joke in data:
            cleanedJoke = {}
            cleanedJoke["openingLine"] = self.clean(joke["title"])
            cleanedJoke["punchline"] = self.clean(joke["body"])
            cleanedJoke["score"] = joke["score"]
            cleanedData.append(cleanedJoke)
        return cleanedData

    def clean(self, joke):
        words = joke.split()
        words = self.punctuate(words, "?")
        words = self.punctuate(words, "!")
        words = self.punctuate(words, ".")
        words = self.punctuate(words, ",")
        # only lowercase first word of sentence?
        words = list(map(lambda x: x.lower(), words))
        return words


    def punctuate(self, words, punctuation):
        punctuationword = None
        for word in words:
            if punctuation in word:
                punctuationword = word
            else:
                punctuationword = None
        if punctuationword is not None:
            words.remove(punctuationword)
            newword = ''.join(ch for ch in punctuationword if ch is not punctuation)
            words.append(newword)
            words.append(punctuation)
        return words

