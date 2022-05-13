import pandas as pd
import pickle
import matplotlib.pyplot as plt
import collections
# import seaborn as sns

#----------------------------------------------------------
# get count of concepts in output
#----------------------------------------------------------

# count number of each concept in my extracted sentences
def getConceptOccurrences(df):
    all_concepts = []
    for index, row in df.iterrows():
        concepts = list(eval(row['forecite_concepts']))
        all_concepts.extend(concepts)

    occurrences = collections.Counter(all_concepts)
    return occurrences

# plot counts of concepts present in my extracted sentences 
def makePlottableDF(occurrences):
    # format data
    scores = []
    counts = []
    concepts = []
    for k in occurrences.keys():
        concepts.append(k)
        scores.append(concept_dict[k])
        counts.append(occurrences[k])

    df_output = pd.DataFrame(list(zip(concepts, scores, counts)), 
    columns =['concept', 'score', 'count']) 
    return df_output

def makePlot(occurrences, filepath):
    # make plottable df
    df = makePlottableDF(occurrences)

    # plot scores vs. counts for each concept
    plot = sns.scatterplot(data=df, x="score", y="count")
    for index, row in df.iterrows():
        plot.text(x=row['score']+0.1, y=row['count']+0.1, s=row['concept'], fontsize=6)

    plot.figure.savefig(filepath)
#-----------------------------

# unpickle concept dictionary
with open('forecite_concept_dict.pickle', 'rb') as handle:
    concept_dict = pickle.load(handle)

#-----------------------------

# data for noun phrase (np) level concepts
df_np = pd.read_csv("./score-threshold-1-0/single-sentence-noun-phrases.csv")
# data for word-level substrings (ss) of noun phrases
df_ss = pd.read_csv("./score-threshold-1-0/single-sentence-all-substrings.csv")

ss_occurrences = getConceptOccurrences(df_ss)
np_occurrences = getConceptOccurrences(df_np)

# print("plotting noun phrase level counts vs. scores...")
# makePlot(np_occurrences, "./score-threshold-1-0/noun-phrase-level-concepts-scatter.png")
# print("done")

# print("plotting substring level counts vs. scores...")
# makePlot(ss_occurrences, "./score-threshold-1-0/substring-level-concepts-scatter.png")
# print("done")

#-----------------------------
# get sentence lengths
df = pd.read_csv("./score-threshold-1-0/single-sentence-min1-noun-phrase.csv")

text = df['sentence']

# plot sentence lengths
seq_len = [len(i.split()) for i in text]
pd.Series(seq_len).hist(bins = 30)
plt.savefig('single-sentence-min1-np-lengths.png')

count = 0
# remove sentences with length < 10 
for index, row in df.iterrows():
    if len(row['sentence'].split()) < 10:
        print(row['sentence'])
        print(row['forecite_concepts'])
        count += 1

        df = df.drop(index)

print(df)
df.to_csv("./score-threshold-1-0/single-sentence-min1-noun-phrase.csv")

