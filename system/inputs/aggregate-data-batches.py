import numpy as np
import pandas as pd
import spacy
import random
from sklearn.model_selection import GroupShuffleSplit
import ast 
import re
import os

#----------------------------------------------------------
# aggregate all batches' 2-sentence, 1-concept instances
#----------------------------------------------------------
# df_all = pd.DataFrame()

# for i in range(0,100):
#     df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows-batch-%d.csv" % i)
#     df_all = df_all.append(df)

# print(df_all)
# df_all = df_all.drop_duplicates(subset=['sentence', 'sentence_original_case', 'forecite_concepts'])
# print(df_all)

# df_all.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows.csv")

#----------------------------------------------------------
# aggregate all demarcated rows
#---------------------------------------------------------
df_demarcated = pd.DataFrame()
directory = "/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/demarcated-batches/"

for filename in os.listdir(directory):
    print(filename)
    df = pd.read_csv(directory + filename)
    df_demarcated = df_demarcated.append(df)

df_demarcated = df_demarcated.drop_duplicates(subset=['paper_id','original_sentence','sentence','forecite_concept'])
print(df_demarcated)


# remove paper_ids that are in the annotated data
df_annotations = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/annotations-round2/union-1sentence-both2sentence/union-multilabel-data-§.csv")
paper_ids_to_exclude = df_annotations['paper_id']
df_demarcated = df_demarcated[~df_demarcated['paper_id'].isin(paper_ids_to_exclude)]

print(df_demarcated)

df_demarcated.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows-demarcated.csv")






