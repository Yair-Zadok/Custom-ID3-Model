import nltk
import numpy as np
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import wordnet
###################################################################################################################################################################################################################

def semantic_similarity(phrase1, phrase2):
    if phrase1 == phrase2:
        return 1

    # Splits the phrases into individual words and tag them with their part of speech
    words1 = pos_tag(phrase1.split())
    words2 = pos_tag(phrase2.split())
    
    # Filters out non-noun words
    nouns1 = [word for word, pos in words1 if pos == 'NN']
    nouns2 = [word for word, pos in words2 if pos == 'NN']
    
    # Calculates the Wu-Palmer Similarity for each pair of nouns
    similarities = []
    for word1 in nouns1:
        for word2 in nouns2:
            word1_synset = wordnet.synset(word1 + '.n.01')  # Specify part of speech as noun
            word2_synset = wordnet.synset(word2 + '.n.01')  # Specify part of speech as noun
            similarity = word1_synset.wup_similarity(word2_synset)
            similarities.append(similarity)
    
    # Calculates the average similarity
    average_similarity = sum(similarities) / (len(similarities) + np.finfo('float').eps) # Epsilon added to avoid devision by zero
    return average_similarity

###################################################################################################################################################################################################################

# Test phrases
phrase1 = "I like ice cream"
phrase2 = "I dislike ice cream"

# Example of how to use: (uncomment below line)
# print("The semantic similarity is:", semantic_similarity(phrase1, phrase2))

###################################################################################################################################################################################################################


