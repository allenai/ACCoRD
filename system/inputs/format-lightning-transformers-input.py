import numpy as np
import pandas as pd
import spacy
import random
from sklearn.model_selection import GroupShuffleSplit
import ast 
import re

setting = "union-1sentence-both2sentence"
# load test set 
df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/annotations-round2/%s/union-multilabel-data-withformattedstatements-§-test.csv" % setting)

sources = []
targets = []

sources_compare = []
sources_isa = []
sources_partof = []
sources_usedfor = []

targets_compare = []
targets_isa = []
targets_partof = []
targets_usedfor = []

separator = "<ENT>"
for index, row in df.iterrows():
    if pd.notna(row['formatted_statement']) and ('[' not in row['formatted_statement']):
        sentence = row['sentence']
        formatted_statement = row['formatted_statement']
        print(sentence)

        matches = re.finditer('\u00a7', sentence)
        # get a list containing only the start indices.
        matches_positions = [match.start() for match in matches]

        # if there are matches
        start = matches_positions[0]
        end = matches_positions[1]
        # get demarcated concept from source sentence
        concept = sentence[start+2:end-1]
        
        # find and demarcate concept in target
        matches = re.finditer(concept, formatted_statement)
        # get a list containing only the start indices.
        matches_positions = [match.start() for match in matches]

        if len(matches_positions) < 1:
            # print(concept)
            # print(formatted_statement)
            continue
        
        start = matches_positions[0]

        demarcated_formatted_statement = formatted_statement[:start] + separator + " " + formatted_statement[start:start+len(concept)] + " " + separator + formatted_statement[start+len(concept):]
        
        # create dataset for all examples
        sources.append(sentence.replace('\u00a7', '<ENT>')) # append new entity demarcated sentence to source list
        targets.append(demarcated_formatted_statement)

        # if 'compare' in row['relation']:
        #     sources_compare.append(sentence.replace('\u00a7', '<ENT>')) # append new entity demarcated sentence to source list
        #     targets_compare.append(demarcated_formatted_statement)
        # if ('is-a' in row['relation']) or ('type-of' in row['relation']):
        #     sources_isa.append(sentence.replace('\u00a7', '<ENT>')) # append new entity demarcated sentence to source list
        #     targets_isa.append(demarcated_formatted_statement)
        # if 'part-of' in row['relation']:
        #     sources_partof.append(sentence.replace('\u00a7', '<ENT>')) # append new entity demarcated sentence to source list
        #     targets_partof.append(demarcated_formatted_statement)
        # if ('used-for' in row['relation']) or ('based-on' in row['relation']):
        #     sources_usedfor.append(sentence.replace('\u00a7', '<ENT>')) # append new entity demarcated sentence to source list
        #     targets_usedfor.append(demarcated_formatted_statement)


dataset = 'train'

# create data for all examples
df_output = pd.DataFrame(list(zip(sources, targets)), 
columns =['source', 'target']) 
df_output.to_json("./%s/%s.json" % (setting, dataset), orient='records', lines=True)


# # create separate datasets for each relation category
# df_output = pd.DataFrame(list(zip(sources_compare, targets_compare)), 
# columns =['source', 'target']) 
# df_output.to_json("./%s/%s-compare.json" % (setting, dataset), orient='records', lines=True)

# df_output = pd.DataFrame(list(zip(sources_isa, targets_isa)), 
# columns =['source', 'target']) 
# df_output.to_json("./%s/%s-isa.json" % (setting, dataset), orient='records', lines=True)

# df_output = pd.DataFrame(list(zip(sources_partof, targets_partof)), 
# columns =['source', 'target']) 
# df_output.to_json("./%s/%s-partof.json" % (setting, dataset), orient='records', lines=True)

# df_output = pd.DataFrame(list(zip(sources_usedfor, targets_usedfor)), 
# columns =['source', 'target']) 
# df_output.to_json("./%s/%s-usedfor.json" % (setting, dataset), orient='records', lines=True)