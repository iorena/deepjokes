import operator

class DataPrepper:
    def __init__(self, data, cutoff):
        self.eos_token = 0
        self.cw_token = 1
        self.sos_token = 2
        self.index = 3
        self.opening_line_length = 0
        self.punchline_length = 0
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
            if len(joke["openingLine"]) > self.opening_line_length:
                self.opening_line_length = len(joke["openingLine"])
            if len(joke["punchline"]) > self.punchline_length:
                self.punchline_length = len(joke["punchline"])
        sortedWords = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
        self.fillerBOW = ["EOS", "CONTENTWORD", "SOS"]
        for i in range(len(sortedWords)):
            if sortedWords[i][1] >= self.cutoff:
                self.fillerBOW.append(sortedWords[i][0])
            else:
                self.contentBOW = [word[0] for word in sortedWords[i:]]
                break
        self.allBOW = words

    def tokenizeData(self):
        data = self.allData
        tokenizedData = []
        for joke in data:
            tokenizedJoke = {}
            tokenizedJoke["openingLine"] = joke["openingLine"]
            tokenizedOpeningLine = list(map(lambda x: self.getToken(x), joke["openingLine"]))
            tokenizedJoke["t_openingLine"], tokenizedJoke["t_openingLineCWs"] = self.separateContentWords(tokenizedOpeningLine)
            tokenizedJoke["punchline"] = joke["punchline"]
            tokenizedPunchline = list(map(lambda x: self.getToken(x), joke["punchline"]))
            tokenizedJoke["t_punchline"], tokenizedJoke["t_punchlineCWs"] = self.separateContentWords(tokenizedPunchline)
            if len(tokenizedJoke["t_punchline"]) == 0:
                print(joke)

            #add EOS tokens
            tokenizedJoke["t_openingLine"].append(self.eos_token)
            tokenizedJoke["t_punchline"].append(self.eos_token)
            tokenizedJoke["t_openingLineCWs"].append(self.eos_token)
            tokenizedJoke["t_punchlineCWs"].append(self.eos_token)

            tokenizedJoke["score"] = joke["score"]
            tokenizedData.append(tokenizedJoke)
        self.allData = tokenizedData

    def countwords(self, words, line):
        max_length = 0
        for word in line:
            if len(word) > max_length:
                max_length = len(word)
            if word in words:
                words[word] = words[word] + 1
            else:
                words[word] = 1
        return max_length

    def cleanup(self, data):
        cleanedData = []
        #Filter out jokes with empty punchline
        for joke in data:
            if len(joke["body"].replace(" ", "")) == 0:
                continue
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
        words = self.punctuate(words, "'s")
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
        for token in line:
            if token < len(self.fillerBOW):
                templatedLine.append(token)
            else:
                templatedLine.append(self.cw_token)
                contentWords.append(token)
        return templatedLine, contentWords

    def getToken(self, word):
        if word in self.fillerBOW:
            return self.fillerBOW.index(word)
        return len(self.fillerBOW) + self.contentBOW.index(word) + 1

    def getCount(self, word):
        if word == "EMPTY":
            return 9001
        return self.allBOW[word]["count"]

    def separateData(self):
        self.dataset = {}
        self.dataset["train"] = self.allData[1:20000]
        self.dataset["test"] = self.allData[20001:]
