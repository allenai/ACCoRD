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
import Levenshtein as lev
import os
import sys


#----------------------------------------------------------
# load data
#----------------------------------------------------------

start = int(sys.argv[1])
end = int(sys.argv[2])

df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows.csv")

df = df[start:end]

# set up sentencizer
nlp = spacy.load("en_core_web_md")
tokenizer = nlp.tokenizer
nlp.add_pipe("sentencizer")
all_stopwords = nlp.Defaults.stop_words
separator = "§"

#----------------------------------------------------------
# demarcate concepts
#----------------------------------------------------------

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

def demarcateConcepts(all_phrase_keys, fc_concept, fc_phrase, sentence):
    # get starting index of this phrase in the sentence
    # get an iterable object containing the start and end indices of each occurrence of pattern in string
    matches = re.finditer(re.escape(str(fc_phrase)), sentence)
    # list comprehension to get a list containing only the start indices.
    matches_positions = [match.start() for match in matches]
    if fc_phrase == "":
        print("sentence had no match with concept = %s" % fc_concept)
        print(all_phrase_keys)
        print(sentence)
        print("")

        txt = "sentence had no concept match"
    elif len(matches_positions) == 1:
        for m in matches_positions:
            start = m
            txt = sentence[:start] + separator + " " + sentence[start:start+len(fc_phrase)] + " " + separator + sentence[start+len(fc_phrase):]
            df.at[index, 'sentence'] = txt

    # if there are multiple occurrences of this forecite concept in the text
    elif len(matches_positions) >= 2:
        for i in range(len(matches_positions)):
            start = matches_positions[i]
            if i == 0:
                txt = sentence[:start] + separator + " " + sentence[start:start+len(fc_phrase)] + " " + separator + sentence[start+len(fc_phrase):]
                sentence = txt # set the modified sentence as the new sentence for the next iteration
            if i == 1:
                start += 4 # add 4 characters to accomodate separator and space x2 that were added in previous round
                txt = sentence[:start] + separator + " " + sentence[start:start+len(fc_phrase)] + " " + separator + sentence[start+len(fc_phrase):]
        df.at[index, 'sentence'] = txt
    
    else:
        txt = "sentence had no concept match"

    return txt

#----------------------------------------------------------
demarcated_sentences = []
lowercase_sentences = []
original_case_sentences = []
concepts = []
paper_ids = []

for index, row in df.iterrows():
    if index % 1000 == 0:
        print("...processing row %d" % index)

    original_case_sentence = row['sentence_original_case']
    sentence = row['sentence']
    paper_id = row['paper_id']
    all_unique_fc_concepts = list(set(list(ast.literal_eval(row['forecite_concepts']))))
    for fc_concept in all_unique_fc_concepts:
        # get variants of forecite concepts for matching
        lemmatized_fc_concept = " ".join([t.lemma_ for t in nlp(fc_concept) if not (t.is_stop)])
        no_punc_fc_concept = re.sub(r'[^\w\s]','', fc_concept)

        doc = nlp(sentence)
        fc_phrase = ""
        all_phrase_keys = []
        for phrase in doc.noun_chunks:
            tokens = [t for t in phrase]
            # process phrase like forecite does
            phrase_key = " ".join([t.lemma_ for t in tokens if not (t.is_stop)])
            all_phrase_keys.append(phrase_key)

            if (fc_concept in phrase_key) or (lemmatized_fc_concept in phrase_key) or (no_punc_fc_concept in re.sub(r'[^\w\s]','', phrase_key)):
                fc_phrase = processPhrase(phrase)
                break

        demarcated_sentence = demarcateConcepts(all_phrase_keys, fc_concept, fc_phrase, sentence)

        # append to lists for dataframe
        demarcated_sentences.append(demarcated_sentence)
        lowercase_sentences.append(sentence)
        original_case_sentences.append(original_case_sentence)
        concepts.append(fc_concept)
        paper_ids.append(paper_id)


# save lists as a df
df_output = pd.DataFrame(list(zip(paper_ids, original_case_sentences, lowercase_sentences, demarcated_sentences, concepts)), 
columns =['paper_id', 'sentence_original_case', 'sentence_lowercase', 'sentence', 'forecite_concept']) 

print(df_output)

ids = df_output['paper_id'].unique()
random.Random(4).shuffle(ids)
df_output = df_output.set_index('paper_id').loc[ids].reset_index()
print(df_output)

df_output.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows-demarcated-%d-%d.csv" % (start, end))
