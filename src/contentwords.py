import jsonloader
import string

#The frequency where a word is considered a content word
CUTOFF_FREQUENCY = 100

class ContentWords:
    def __init__(self, minimum_karma=5):
        loader = jsonloader.Jsonloader(minimum_karma)
        print(len(loader.data))
        self.index = 0
        self.createBOW(loader.data)
        print(self.contentBOW)
        print(self.fillerBOW)

    def createBOW(self, data):
        words = {}
        for line in data:
            title = self.cleanup(line["title"])
            body = self.cleanup(line["body"])
            self.countwords(words, title)
            self.countwords(words, body)
        self.contentBOW = {word: words[word] for word in words if words[word]["count"] >= CUTOFF_FREQUENCY}
        self.fillerBOW = {word: words[word] for word in words if words[word]["count"] < CUTOFF_FREQUENCY}

    def countwords(self, words, line):
        for word in line:
            if word in words:
                words[word]["count"] = words[word]["count"] + 1
            else:
                words[word] = {}
                words[word]["count"] = 1
                words[word]["id"] = self.index
                self.index += 1

    def cleanup(self, line):
        words = line.split()
        words = self.punctuate(words, "?")
        words = self.punctuate(words, "!")
        words = self.punctuate(words, ".")
        words = self.punctuate(words, ",")
        # only lowercase first word of sentence?
        words = list(map(lambda x: x.lower(), words))
        #words = map(lambda x: ''.join(ch for ch in x if ch not in set([",", "."])), words)
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


contentwords = ContentWords(1)

