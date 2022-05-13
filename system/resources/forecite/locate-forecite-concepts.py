import math 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
import matplotlib.pyplot as plt
from transformers import *
import matplotlib.ticker as mticker
import spacy
import ast
import re
import random

def processPhrase(phrase):
  if len(phrase.text.split()) > 1:
    p = phrase.text.split()
    if p[0] in all_stopwords: # if first word is a stop word
      phrase_no_stopwords = " ".join(p[1:])
    elif p[-1] in all_stopwords: # if last word is a stop word
      phrase_no_stopwords = " ".join(p[:-1])
    else: # if neither is a stop word
      phrase_no_stopwords = phrase.text

  # if there is only one word in this phrase, just append the text version of it
  else: phrase_no_stopwords = phrase.text

  return phrase_no_stopwords


def main(sentence, separator, forecite_concept):
    doc = nlp(sentence)
    fc_phrase = ""
    for phrase in doc.noun_chunks:
        tokens = [t for t in phrase]
        phrase_key = " ".join([t.lemma_ for t in tokens if not (t.is_stop)])

        if phrase_key == forecite_concept:
            fc_phrase = processPhrase(phrase)
            break

    # get starting index of this phrase in the sentence
    # get an iterable object containing the start and end indices of each occurrence of pattern in string
    matches = re.finditer(str(fc_phrase), sentence)
    # get a list containing only the start indices.
    matches_positions = [match.start() for match in matches]
    # if there are matches
    if len(matches_positions) > 0:
      start = matches_positions[0]
      txt = sentence[:start] + separator + " " + sentence[start:start+len(fc_phrase)] + " " + separator + sentence[start+len(fc_phrase):]
    else:
        txt = "concept not found"

    return txt

#----------------------------------------------------------
# set up sentencizer
nlp = spacy.load("en_core_web_md")
tokenizer = nlp.tokenizer
nlp.add_pipe("sentencizer")
all_stopwords = nlp.Defaults.stop_words
separator = "§"

# # ex 1
# sentence = "neural networks with long short-term memory (lstm), which have emerged as effective and scalable model for several learning problems related to sequential data (e.g., handwriting recognition, speech recognition, human activity recognition and traffic prediction), and it does not suffer from the effect of the vanishing or exploding gradient problem as simple recurrent networks do [1] ."
# forecite_concept = "long short - term memory"
# print(main(sentence, separator, forecite_concept))

# # ex 2
# sentence = "person reidentification is a task of recognizing person based on appearance matching under different camera views."
# forecite_concept = "different camera view"
# print(main(sentence, separator, forecite_concept))

# # ex 3
# sentence = "according to google, the tpu can compute neural networks up to 30x faster and up to 80x more power efficient than cpu's or gpu's performing similar applications [6] . the tpu excels because its hardware processing flow is specifically adapted to the inference problem it solves."
# forecite_concept = "tpu"
# print(main(sentence, separator, forecite_concept))

df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/annotations-round2/union-test-from-heddex-results.csv")

demarcated_sentences = []
for index, row in df.iterrows():
  demarcated_sentence = main(row['sentence'], separator, row['forecite_concept'])
  print(row['sentence'])
  print(demarcated_sentence)
  print()
  demarcated_sentences.append(demarcated_sentence)

df['sentence'] = demarcated_sentences
print(df)
df.to_csv("/net/nfs2.s2-research/soniam/concept-rel/annotations-round2/union-test-from-heddex-results-%s.csv" % separator)

