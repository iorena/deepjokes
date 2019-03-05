import json

class Jsonloader:

    def __init__(self, minimum_karma):
        with open("../data/reddit_jokes.json") as infile:
            self.data = json.load(infile)
            print("Loaded", len(self.data), "rows of data")

        #Filter to keep only jokes with one opening line and punchline and format
        self.data = list(filter(lambda x: 1 < len(x["body"]) < 30, self.data))

        #Filter out jokes with karma too low
        self.data = list(filter(lambda x: x["score"] >= minimum_karma, self.data))

        print("Data size after filtering:", len(self.data))


        with open("../data/filtered_data.json", "w") as outfile:
            json.dump(self.data, outfile)
