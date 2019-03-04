CUTOFF_FREQUENCY = 100

class JokeModel:
    def __init__(self, data):
        for joke in data.dataset["train"]:
            for word in joke["openingLine"]:
                print(word)
            print("###")
