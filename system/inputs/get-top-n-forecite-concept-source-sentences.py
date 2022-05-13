import pickle
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from operator import itemgetter
import numpy as np
import re

# load final df of sentences to select concept-specific data from
df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/scibert-multilabel-binary-predictions-all-s2orc-with-forecite-concepts.csv")
print(df.keys())
print(len(set(df['paper_id'].to_list())))
#--------------------------------------------------------
# unpickle concept dictionary
with open('/net/nfs2.s2-research/soniam/concept-rel/resources/forecite/forecite_concept_dict.pickle', 'rb') as handle:
    concept_dict = pickle.load(handle)
    print("...unpickled forecite concept dictionary")

# Initialize N
N = 50
# N largest values in dictionary
# Using sorted() + itemgetter() + items()
res = dict(sorted(concept_dict.items(), key = itemgetter(1), reverse = True)[:N])
# printing result
print("The top N value pairs are " + str(res))
concept_list = list(res.keys())
# print(concept_list)
#--------------------------------------------------------
selected_nlp_concepts = ['adversarial training', 'beam search', 'bert', 'elmo', 'gpt', 'glove', 'word2vec', 'resnet', 'domain shift', 'ulmfit', 'newsqa', 'squad', 'random forest', 'imagenet', 'lstm', 'roberta', 'variational autoencoder', 'dropout', 'fasttext', 'hierarchical softmax', 'distant supervision']
#--------------------------------------------------------
df_nlp_concepts = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/nlp_concepts.csv")
nlp_concept_scores = []
for index, row in df_nlp_concepts.iterrows():
    # get forecite score for each nlp concept
    nlp_concept_scores.append(concept_dict[row['concept']])

df_nlp_concepts['score'] = nlp_concept_scores
df_nlp_concepts = df_nlp_concepts[~df_nlp_concepts.concept.isin(['bert model', 'pre - train bert model', 'pre - train bert', 'moses', 'cho et al', 'dagan', 'yarowsky', 'hochreiter', 'turney', 'ney', 'och', 'grosz', 'steedman', 'well translation'])]

# # top 150 nlp concepts by score
# top150_nlp_concepts = df_nlp_concepts.sort_values(by='score', ascending=False)[:150]['concept'].tolist()
# top 150 nlp concepts by num_papers
top150_nlp_concepts = df_nlp_concepts.sort_values(by='num_papers', ascending=False)[:150]['concept'].tolist()
print(top150_nlp_concepts)
#--------------------------------------------------------
num_sents = 0
# get predictions for all rows for a concept
for concept in top150_nlp_concepts:
    df_concept = df[df['forecite_concept']==concept]
    print(concept, len(df_concept))
    num_sents += len(df_concept)

    if len(df_concept) == 0:
        continue

    # process multilabel pred scores
    # indices for old 4class correspond to ['compare' 'is-a' 'part-of' 'used-for']
    # indices for new 3class correspond to ['compare' 'contrast' 'isa']
    all_preds = []
    for index, row in df_concept.iterrows():
        temp = row['multilabel_pred'].strip("[]").split()
        preds = []
        for t in temp:
            preds.append(float(t))

        all_preds.append(preds)

    all_preds = np.array(all_preds)

    # convert multilabel pred scores to a label by selecting the max pred score
    result = np.argmax(all_preds, axis=1)   
    max = np.max(all_preds, axis=1)   
    df_concept['index_of_max_multilabel_pred'] = result
    df_concept['max_multilabel_pred_score'] = max

    # # convert multilabel pred scores to a label by taking any score above 0 (equals >0.5 probability)
    # bool_ind = all_preds > 0
    # print(len(bool_ind[:,1]))
    # print(np.any(bool_ind[:,1]))

    df_concept['compare_pred_score'] = all_preds[:,0]
    df_concept['isa_pred_score'] = all_preds[:,1]
    df_concept['partof_pred_score'] = all_preds[:,2]
    df_concept['usedfor_pred_score'] = all_preds[:,3]

    # try to find "contrast" label examples within "compare" predictions using heuristics
    contrast_words = ['unlike', 'alternative to', 'alternatively', 'conversely', 'than', 'contrast', 'compared to', 'in comparison', 'instead', 'whereas', 'while', 'except', 'previous', 'different from', 'different to', 'differs', 'extends', 'extension']
    possible_contrast = []
    for index, row in df_concept.iterrows():
        value = -1
        if (row['compare_pred_score'] > 0):
            for word in contrast_words:
                if re.search(r'\b' + word + r'\b', row['original_sentence']):
                    value = row['compare_pred_score']
        
        possible_contrast.append(value)
        if value>0:
            print(row['compare_pred_score'], row['original_sentence'])
    df_concept['contrast_pred_score'] = possible_contrast

    df_concept.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/top-n-forecite-concept-source-sentences/v1-4class/nlp-concepts/scibert-multilabel-binary-predictions-all-s2orc-with-forecite-%s.csv" % concept)

print(num_sents)