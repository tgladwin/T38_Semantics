import spacy

nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# Observations

# The program identifies that there is a link between an apple and a banana being a fruit
# From here it shows that there is a small link between an apple and a monkey based on the priciple that a monkey
# will eat a banana, a banana is a fruit, an apple is also a fruit and therefore a link between monkey -> apple appears
# It does not matter which words precedes the other in the comparison e.g. monkey -> banana OR banana -> monkey

# When comparing the sentences, the program identifies a strong similarity between line 16 and line 18. This link is
# made because both lines contain the word car, have a similar amount of syllables. However, line 19 also contains the
# word car, but has a much lower similarity. I believe that this happens because the sentence is grammatically incorrect
# and lacks some logical sense.

tokens = nlp('car truck petrol lemonade ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# In the above example, there are strong links made between car and truck as they are both forms of transport.
# There is a weak link between petrol and lemonade, however I thought this would be stronger as they are both liquids.
# There is a medium link between car & petrol, and truck & petrol. I believe that this is close to a 50% link but not
# quite as whilst petrol and diesel are the most common forms of fuel in vehicles (and therefore could be 50/50) there
# are alternative fuels available that make up a small share

# When running the example file with the more simple language model, significantly weaker links are found between
# entries when compared to the md model. 
