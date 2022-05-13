import pickle
import pandas as pd
from operator import itemgetter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import exists

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
# print("The top N value pairs are " + str(res))
concept_list = list(res.keys())
# print(concept_list)
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

selected_nlp_concepts = ['adversarial training', 'beam search', 'bert', 'elmo', 'gpt', 'glove', 'word2vec', 'resnet', 'domain shift', 'ulmfit', 'newsqa', 'squad', 'random forest', 'imagenet', 'lstm', 'roberta', 'variational autoencoder', 'dropout', 'fasttext', 'hierarchical softmax', 'distant supervision']
relations = ['compare','isa']

#--------------------------------------------------------
# 1) PLOT NUMBER OF DESCRIPTIONS BEFORE RANKING FILTERING
#--------------------------------------------------------
description_counts = []
relation_counts = []
concept_counts = []
isa_counts = []
compare_counts = []

version = "v1-4class/nlp-concepts"

for concept in top150_nlp_concepts:
    df_concept = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/top-n-forecite-concept-source-sentences/%s/scibert-multilabel-binary-predictions-all-s2orc-with-forecite-%s.csv" % (version, concept))
    df_concept = df_concept.loc[:, ~df_concept.columns.str.contains('^Unnamed')]
    
    if len(df_concept) == 0:
        continue

    for index, relation in enumerate(relations):
        df_concept_relation = df_concept[df_concept['%s_pred_score' % relation]>0]

        # add to df to plot
        concept_counts.append(concept)
        relation_counts.append(relation)
        description_counts.append(len(df_concept_relation))

        if relation == "compare":
            compare_counts.append(len(df_concept_relation))
        elif relation == "isa":
            isa_counts.append(len(df_concept_relation))


print(sum(isa_counts)/len(isa_counts))
print(sum(compare_counts)/len(compare_counts))
# plot for first N concepts (by number of papers)
num_concepts = 50
df_counts = pd.DataFrame(list(zip(concept_counts, relation_counts, description_counts)), columns =['concept', 'relation', 'num_descriptions'])
df_counts_isa = df_counts[df_counts['relation']=="isa"].sort_values(by='num_descriptions', ascending=False)[:num_concepts]
df_counts_compare = df_counts[df_counts['relation']=="compare"].sort_values(by='num_descriptions', ascending=False)[:num_concepts]

# # plot compare
# sns.catplot(data=df_counts_compare, x="concept", y="num_descriptions", col="relation", kind="bar", height=5, aspect=3)
# plt.xticks(rotation='vertical')
# plt.savefig("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/v1-4class/plots/description-counts-compare.pdf", bbox_inches="tight")

# # plot isa
# sns.catplot(data=df_counts_isa, x="concept", y="num_descriptions", col="relation", kind="bar", height=5, aspect=3)
# plt.xticks(rotation='vertical')
# plt.savefig("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/v1-4class/plots/description-counts-isa.pdf", bbox_inches="tight")

#--------------------------------------------------------
# GET STATS FOR NUMBER OF CONCEPT Bs per CONCEPT A
#--------------------------------------------------------

version = "v1-4class"
concept_set = "nlp-concepts"
relations = ['compare', 'isa']

concepta_conceptb_counts = []
relation_conceptb_counts = []
conceptb_counts = []
isa_counts = []
compare_counts = []

for concept in top150_nlp_concepts:
    for index, relation in enumerate(relations):
        file_path = "/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/ranked-filtered/%s/%s-%s.csv" % (version, concept_set, concept, relation)
        if exists(file_path):
            df_ranked_filtered_concept_relation = pd.read_csv(file_path)
        else:
            continue
        # skip if file doesn't have any descriptions
        if len(df_ranked_filtered_concept_relation) == 0:
            continue

        concepta_conceptb_counts.append(concept)
        relation_conceptb_counts.append(relation)
        conceptb_counts.append(len(df_ranked_filtered_concept_relation))

        if relation == "compare":
            compare_counts.append(len(df_ranked_filtered_concept_relation))
        elif relation == "isa":
            isa_counts.append(len(df_ranked_filtered_concept_relation))

print(sum(isa_counts)/len(isa_counts))
print(sum(compare_counts)/len(compare_counts))
num_concepts = 150
df_counts = pd.DataFrame(list(zip(concepta_conceptb_counts, relation_conceptb_counts, conceptb_counts)), columns =['concept', 'relation', 'num_concept_bs'])
df_counts_isa = df_counts[df_counts['relation']=="isa"].sort_values(by='num_concept_bs', ascending=False)[:num_concepts]
df_counts_compare = df_counts[df_counts['relation']=="compare"].sort_values(by='num_concept_bs', ascending=False)[:num_concepts]

# plot compare
fig, axs = plt.subplots(1, 2, figsize=(4, 3))
plt.setp(axs, ylim=(0,41))

sns.histplot(data=df_counts_isa, x="num_concept_bs", color="darkcyan", ax=axs[0])
sns.histplot(data=df_counts_compare, x="num_concept_bs", color="darkorange", ax=axs[1])

# sns.catplot(x="num_concept_bs", col="relation", data=df_counts, kind="count", height=4, aspect=.7);
# sns.histplot(data=df_counts, x="num_concept_bs", hue="relation")
plt.savefig("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/v1-4class/plots/conceptb-counts-histogram.pdf", bbox_inches="tight")