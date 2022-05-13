import pandas as pd
import ast
import numpy as np

experiment = "best-params-all-s2orc"
df_preds = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/scibert-multilabel-classification/run-best-model/predictions/scibert-weightedBCE-cls/%s/seed=1-epochs=10-lr=0.000020-bs=32-%s.csv" % (experiment, experiment))

df_train_val = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/annotations-round2/union-1sentence-both2sentence/union-multilabel-data-§.csv")
# we don't want to consider the paper_ids that were in the training and val sets
paper_ids_to_exclude = df_train_val['paper_id']

df_test = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows-demarcated.csv")
# df_test_undemarcated = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows.csv")

# get paper ids and forecite_concepts from test data used in the best scibert-multilabel run
paper_ids = df_test['paper_id']
forecite_concepts = df_test['forecite_concept']
lowercase_sentences = df_test['original_sentence']


# add these paper_ids and forecite_concepts as a column to the predictions df
df_preds['paper_id'] = paper_ids
df_preds['forecite_concept'] = forecite_concepts
df_preds['sentence_lowercase'] = lowercase_sentences


# print(len(df_preds))
# # remove paper_ids in train/val set from predictions
# df_preds = df_preds[~df_preds['paper_id'].isin(paper_ids_to_exclude)]
# print(len(df_preds))

# convert string predictions to list
demarcated_sentences = df_preds['sentence'].tolist()
paper_ids = df_preds['paper_id'].tolist()
all_preds = []
for index, row in df_preds.iterrows():
    temp = row['scibert_pred'].strip("[]").split()
    preds = []
    for t in temp:
        preds.append(float(t))

    all_preds.append(preds)

all_preds = np.array(all_preds)

def getSortedPredictions(all_preds, column_to_sort, category):
    # sort on column_to_sort of all predictions and then reverse the array to get max -> min
    argsort_indices = np.argsort(all_preds[:,column_to_sort])[::-1]
    sorted_demarcated_sentences = np.array(demarcated_sentences)[argsort_indices]
    sorted_lowercase_sentences = np.array(lowercase_sentences)[argsort_indices]
    sorted_paper_ids = np.array(paper_ids)[argsort_indices]
    sorted_forecite_concepts = np.array(forecite_concepts)[argsort_indices]
    sorted_preds = all_preds[argsort_indices]

    # top20_sentences = sorted_sentences[:20]
    # top20_preds = sorted_preds[:20]

    df_output = pd.DataFrame(list(zip(sorted_paper_ids, sorted_lowercase_sentences, sorted_demarcated_sentences, sorted_forecite_concepts, sorted_preds)), 
    columns =['%s_paper_ids' % category, '%s_lowercase_sentences' % category, '%s_demarcated_sentences' % category, '%s_forecite_concepts' % category, '%s_preds' % category]) 
    print(df_output)

    df_output.to_csv("/net/nfs2.s2-research/soniam/concept-rel/scibert-multilabel-classification/run-best-model/predictions/scibert-weightedBCE-cls/%s/sorted-predictions-seed=1-epochs=10-lr=0.000020-bs=32-%s.csv" % (experiment, category))

    return sorted_paper_ids, sorted_lowercase_sentences, sorted_demarcated_sentences, sorted_forecite_concepts, sorted_preds

# ['compare' 'is-a' 'part-of' 'used-for']
# sorted_compare_paper_ids, sorted_compare_lowercase_sentences, sorted_compare_demarcated_sentences, sorted_compare_forecite_concepts, sorted_compare_preds = getSortedPredictions(all_preds, 0, 'compare') # compare = column 0
# sorted_partof_paper_ids, sorted_partof_lowercase_sentences, sorted_partof_demarcated_sentences, sorted_partof_forecite_concepts, sorted_partof_preds = getSortedPredictions(all_preds, 2, 'partof') # part-of = column 2
# sorted_usedfor_paper_ids, sorted_usedfor_lowercase_sentences, sorted_usedfor_demarcated_sentences, sorted_usedfor_forecite_concepts, sorted_usedfor_preds = getSortedPredictions(all_preds, 3, 'usedfor') # used-for = column 3


# getSortedPredictions(all_preds, 0, 'compare') # compare = column 0
getSortedPredictions(all_preds, 2, 'partof') # part-of = column 2
getSortedPredictions(all_preds, 3, 'usedfor') # used-for = column 3


# df_output = pd.DataFrame(list(zip(sorted_compare_paper_ids, sorted_compare_sentences, sorted_compare_preds, sorted_partof_paper_ids, sorted_partof_sentences, sorted_partof_preds, sorted_usedfor_paper_ids, sorted_usedfor_sentences, sorted_usedfor_preds)), 
#     columns =['compare_paper_id', 'compare_sentence', 'compare_pred', 'partof_paper_id', 'partof_sentence', 'partof_pred', 'usedfor_paper_id', 'usedfor_sentence', 'usedfor_pred']) 
# print(df_output)

# df_output.to_csv("/net/nfs2.s2-research/soniam/concept-rel/scibert-multilabel-classification/run-best-model/predictions/scibert-weightedBCE-cls/%s/sorted-predictions-seed=1-epochs=1-lr=0.000020-bs=32-%s.csv" % (experiment, experiment))


