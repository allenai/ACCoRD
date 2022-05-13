import pandas as pd
import random

#------------------------------------------------------------------------------------------------
# filter output from top 150 NLP concepts to only those concepts with 3+ descriptions
# format demo data to only include [concept B] [elaboration]
# INPUT = ../ranked-filtered/nlp-concepts/top-150-nlp-concepts.csv
# OUTPUT = ../demo-data/top-nlp-concepts-with-gt-2-descriptions.csv and "".json
#     or = ../error-analysis-data/top-nlp-concepts-with-gt-2-descriptions.csv and "".json
#------------------------------------------------------------------------------------------------

data_version = 'demo-data'

df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/v1-4class/ranked-filtered/nlp-concepts/top-150-nlp-concepts.csv")

print(df.keys())
# # select rows for error analysis spreadsheet
# df = df[['paper_id','forecite_concept', 'relation', 'concept_b', 'original_sentence', 'gpt_generation']]

# make paper_id into link
df['paper_id'] = 'api.semanticscholar.org/CorpusId:' + df['paper_id'].astype(str)
print(df)

# select rows where forecite_concept has 2+ rows
v = df['forecite_concept'].value_counts()
df = df[df['forecite_concept'].isin(v.index[v.gt(2)])]

print("number of forecite concepts with at least 3 entries: %d" % len(df['forecite_concept'].value_counts().to_list()))
print(df['forecite_concept'].value_counts())

# remove forecite concept from gpt generation for demo
df['gpt_generation_demo'] = df.apply(lambda L: L['gpt_generation'].replace(L['forecite_concept'] + " ", ''), axis=1)
df['gpt_generation_demo'] = df.apply(lambda L: L['gpt_generation_demo'].replace('is a ', ''), axis=1)
df['gpt_generation_demo'] = df.apply(lambda L: L['gpt_generation_demo'].replace('is like ', ''), axis=1)
print(df['gpt_generation_demo'])

ids = df['forecite_concept'].unique()
random.Random(42).shuffle(ids)
df = df.set_index('forecite_concept').loc[ids].reset_index()

print(df)

# save filtered data for demo and error analysis
df.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/v1-4class/%s/top-nlp-concepts-with-gt-2-descriptions.csv" % data_version)
df.to_json("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/v1-4class/%s/top-nlp-concepts-with-gt-2-descriptions.json" % data_version, orient="records")