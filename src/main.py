import jsonloader
import dataprepper
import jokemodel

#Minimum reddit karma for a joke to be included in the dataset
MINIMUM_KARMA = 5
#The frequency where a word is considered a content word
CUTOFF_FREQUENCY = 100

loader = jsonloader.Jsonloader(MINIMUM_KARMA)
data = dataprepper.DataPrepper(loader.data, CUTOFF_FREQUENCY)

model = jokemodel.JokeModel(data)


