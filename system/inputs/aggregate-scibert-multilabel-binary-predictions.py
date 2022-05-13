import pandas as pd
from ast import literal_eval
#--------------------------------------------------------
# AGGREGATE SCIBERT MULTILABEL AND BINARY PREDICTIONS
#--------------------------------------------------------

# get df of all positive predictions from scibert
df_binary_positive = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/bart-lightning-transformers/scibert-positive-predictions-all-s2orc-with-concepta.csv")
print(df_binary_positive)

# get df of all rows with forecite concepts
df_binary_positive_with_concepts = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows-demarcated.csv")
print(df_binary_positive_with_concepts)
print(df_binary_positive_with_concepts.keys())

# get df of all multilabel predictions from scibert
df_multilabel = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/scibert-multilabel-classification/run-best-model-corrected-with-contrast-class/predictions/scibert-3class-weightedBCE-cls/best-params-all-s2orc/seed=42-epochs=10-lr=0.000050-bs=32-best-params-all-s2orc.csv")
print(df_multilabel)

# select the rows from the multilabel dataset that have the same sentences as the binary label dataset
df_multilabel_positive = df_multilabel.merge(df_binary_positive, on=["sentence"])
print(df_multilabel_positive.keys())
print(df_multilabel_positive)

df_multilabel_positive_with_forecite_concepts = pd.merge(df_binary_positive_with_concepts, df_multilabel_positive, on="sentence", how="outer")
# remove rows where forecite_concept is nan
df_multilabel_positive_with_forecite_concepts = df_multilabel_positive_with_forecite_concepts.dropna()
print(df_multilabel_positive_with_forecite_concepts.keys())
# Select the ones you want
df_multilabel_positive_with_forecite_concepts = df_multilabel_positive_with_forecite_concepts[['paper_id', 'original_sentence','sentence', 'forecite_concept', 'scibert_pred_x','scibert_pred_y','bart_sentences', 'concept_a']]

print(df_multilabel_positive_with_forecite_concepts)
print(df_multilabel_positive_with_forecite_concepts.keys())

print(df_multilabel_positive_with_forecite_concepts['scibert_pred_x'])
print(df_multilabel_positive_with_forecite_concepts['scibert_pred_y'])

df_multilabel_positive_with_forecite_concepts = df_multilabel_positive_with_forecite_concepts.rename(columns={"scibert_pred_x": "multilabel_pred", "scibert_pred_y": "binary_pred"})
# df_multilabel_positive_with_forecite_concepts['multilabel_pred'] = pd.eval(df_multilabel_positive_with_forecite_concepts['multilabel_pred'])
# print(type(df_multilabel_positive_with_forecite_concepts.at[0, 'multilabel_pred']))

df_multilabel_positive_with_forecite_concepts.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/scibert-multilabel-binary-predictions-all-s2orc-with-forecite-concepts-revised-3class.csv")