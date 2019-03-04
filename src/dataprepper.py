class DataPrepper:
    def __init__(self, data, cutoff):
        self.index = 0
        self.cutoff = cutoff
        self.allData = self.cleanup(data)
        self.createBOW()
        self.tokenizeData()
        self.separateData()

    def createBOW(self):
        words = {}
        for joke in self.allData:
            self.countwords(words, joke["openingLine"])
            self.countwords(words, joke["punchline"])
            self.cw_token = self.index + 1
        self.allBOW = words
        self.contentBOW = {word: words[word] for word in words if words[word]["count"] < self.cutoff}
        self.fillerBOW = {word: words[word] for word in words if words[word]["count"] >= self.cutoff}

    def separateData(self):
        self.dataset = {}
        self.dataset["train"] = self.allData[1:20000]
        self.dataset["test"] = self.allData[20001:]

    def tokenizeData(self):
        data = self.allData
        tokenizedData = []
        for joke in data:
            tokenizedJoke = {}
            openingLine, contentwords = self.separateContentWords(joke["openingLine"])
            tokenizedJoke["openingLine"] = openingLine
            tokenizedJoke["openingLineCWs"] = contentwords
            tokenizedJoke["t_openingLine"] = map(lambda x: self.getIndex(x), openingLine)
            tokenizedJoke["t_openingLineCWs"] = map(lambda x: self.getIndex(x), contentwords)
            punchline, contentwords = self.separateContentWords(joke["punchline"])
            tokenizedJoke["punchline"] = punchline
            tokenizedJoke["punchlineCWs"] = contentwords
            tokenizedJoke["t_punchline"] = map(lambda x: self.getIndex(x), punchline)
            tokenizedJoke["t_punchlineCWs"] = map(lambda x: self.getIndex(x), contentwords)
            tokenizedJoke["score"] = joke["score"]
            tokenizedData.append(tokenizedJoke)
        self.allData = tokenizedData

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

    def separateContentWords(self, line):
        templatedLine = []
        contentWords = []
        for word in line:
            if self.allBOW[word]["count"] > self.cutoff:
                templatedLine.append(word)
            else:
                templatedLine.append("CONTENTWORD")
                contentWords.append(word)
        return templatedLine, contentWords

    def getIndex(self, word):
        if word == "CONTENTWORD":
            return self.cw_token
        return self.allBOW[word]["id"]


