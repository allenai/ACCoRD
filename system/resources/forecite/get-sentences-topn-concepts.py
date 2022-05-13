import pickle
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import re
from collections import Counter
import random
import sys

#----------------------------------------------------------
# process ForeCite concepts
#----------------------------------------------------------

# unpickle concept dictionary
with open('forecite_concept_dict.pickle', 'rb') as handle:
    concept_dict = pickle.load(handle)
    print("unpickled forecite concept dictionary")

# print(concept_dict["k - near neighbor"]) # found => lemmatization is correct
# print(concept_dict["k-nearest neighbor"]) # not found

# # plot histogram of scores
# n, bins, patches = plt.hist(concept_dict.values(), 10, density=True, facecolor='g', alpha=0.75)
# print(len(set(concept_dict.values())))
# print(max(concept_dict.values()))
# print(min(concept_dict.values()))
# plt.xlabel('ForeCite score')
# plt.ylabel('Count')
# plt.savefig('scores-histogram.png')

# get concepts with score above a certain threshold
score_threshold = 1.0
concepts_above_threshold = [k for k,v in concept_dict.items() if float(v) >= score_threshold]
concept_set = set(concepts_above_threshold)
print("%d concepts with score above %2f" % (len(concept_set), score_threshold))

# max_length = 0
# max_word = ""
# lens = []
# for c in concepts_above_threshold:
#     c = c.split()
#     lens.append(len(c))
#     if len(c) > max_length:
#         max_length = len(c)
#         max_word = c

# print("max length of concepts = %d" % (max_length))
# print(max_word)
# print()

# plt.hist(lens)
# plt.savefig("concept-length-hist.png")

#----------------------------------------------------------
# helper functions
#----------------------------------------------------------
def getPhraseSubstrings(doc):
    # get spacy noun chunks for each sentence
    phrase_keys = []
    for phrase in doc.noun_chunks:
        # iterate over tokens and get all word substrings
        for i in range(len(phrase)):
            for j in range(i, len(phrase)+1):
                tokens = [t for t in phrase[i:j]]
                lemmatized = [t.lemma_ for t in tokens if not (t.is_stop)] # lemmatize
                phrase_key = " ".join(lemmatized)
                phrase_keys.append(phrase_key)

    return phrase_keys

def getPhrases(doc):
    phrase_keys = []
    for phrase in doc.noun_chunks:
        tokens = [t for t in phrase]
        phrase_key = " ".join([t.lemma_ for t in tokens if not (t.is_stop)])
        phrase_keys.append(phrase_key)

    return phrase_keys

#----------------------------------------------------------
# get sentences with two or more concepts 
df_text_0 = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/s2orc/text-batch-id-0.csv")
df_text_1 = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/s2orc/text-batch-id-1.csv")
df_text_2 = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/s2orc/text-batch-id-2.csv")
df_text_3 = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/s2orc/text-batch-id-3.csv")
df_text_4 = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/s2orc/text-batch-id-4.csv")

df = pd.concat([df_text_1, df_text_2, df_text_3, df_text_4])

print("total rows across all batches = %d" % len(df))

# set up sentencizer
nlp = spacy.load("en_core_web_md")
tokenizer = nlp.tokenizer
nlp.add_pipe("sentencizer")

# test_id = 27044448
# df_paper = df.loc[df['paper_id'] == test_id]

#----------------------------------------------------------
# 2 sentence: only one sentence has a forecite concept
#----------------------------------------------------------
# # get paper ids from final 1- and 2-sentence annotations and don't include those in this dataset
# df1 = pd.read_csv("../../annotations-round2/single-sentence-final-§.csv")
# df2 = pd.read_csv("../../annotations-round2/2-sentence-final-§.csv")
# df1_paper_ids = set(df1['paper_id'])
# df2_paper_ids = set(df2['paper_id'])
# all_annotated_paper_ids = df1_paper_ids | df2_paper_ids

def twoSentenceOneConcept(batch_id):
    # iterate over rows in df (paragraph level)
    num_sents = 0
    paper_ids = []
    sentences = []
    sentences_cased = []
    concepts = []

    df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/s2orc/text-batch-id-%d.csv" % batch_id)

    print("%d rows in batch %d" % (len(df), batch_id))
    
    for index, row in df.iterrows():
        # progress update
        if index % 100 == 0: print('...processing row %d' % (index))

        # # if this paper_id doesn't already have an annotation
        # if row['paper_id'] not in all_annotated_paper_ids:
        # check that row is not na
        if pd.notna(row['text']):
            doc = nlp(row['text'])

            sents = list(doc.sents)
            # iterate over sentences
            for s in range(1, len(sents)-1):

                num_sents += 1
                count = 0

                previous_sent =  sents[s-1].text.lower()
                current_sent = sents[s].text.lower()
                next_sent = sents[s+1].text.lower()

                previous_sent_cased =  sents[s-1].text
                current_sent_cased = sents[s].text
                next_sent_cased = sents[s+1].text



                # if the context sentences are too short to be good, just continue
                if (len(previous_sent) < 5) or (len(next_sent) < 5):
                    continue
                
                else:
                    # # get spacy noun chunks
                    # curr_sent_concepts = set(getPhrases(nlp(current_sent)))

                    # # get intersection of sentence concepts and all concepts
                    # curr_intersection = concept_set.intersection(curr_sent_concepts)

                    # generate random number to pick whether prev or next sentence gets appended as context
                    num = random.randint(1, 2)
                    if num == 1:
                        concat = previous_sent + " " + current_sent
                        concat_cased = previous_sent_cased + " " + current_sent_cased
                    if num == 2:
                        concat = current_sent + " " + next_sent
                        concat_cased = current_sent_cased + " " + next_sent_cased

                    # get spacy noun chunks
                    concat_sent_concepts = set(getPhrases(nlp(concat)))
                    # get intersection of sentence concepts and all concepts
                    concat_intersection = concept_set.intersection(concat_sent_concepts)

                    if len(concat_intersection) > 0:
                        paper_ids.append(row['paper_id'])
                        sentences.append(concat) 
                        sentences_cased.append(concat_cased)               
                        concepts.append(concat_intersection)                    

        # if na skip over this row
        else: continue

    print("%d sentences in %d rows" % (num_sents, index))

    # make final output df
    df_output = pd.DataFrame(list(zip(paper_ids, sentences, sentences_cased, concepts)), 
    columns =['paper_id', 'sentence', 'sentence_original_case', 'forecite_concepts']) 
    print(df_output)
    df_output.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows-batch-%d.csv" % batch_id)

id = int(sys.argv[1])
twoSentenceOneConcept(id)

# # make final output df for cased version-
# df_output = pd.DataFrame(list(zip(paper_ids, sentences_cased, concepts)), 
# columns =['paper_id', 'sentence', 'forecite_concepts']) 
# print(df_output)
# df_output.to_csv("./score-threshold-1-0/2-sentence-only-one-sent-has-fc-concept-originalcase.csv")

# #----------------------------------------------------------
# # 3 sentence: noun phrase level, middle sentence has forecite concept
# #----------------------------------------------------------
# # get paper ids from final 1- and 2-sentence annotations and don't include those in this dataset
# df1 = pd.read_csv("../../annotations-round2/single-sentence-final-§.csv")
# df2 = pd.read_csv("../../annotations-round2/2-sentence-final-§.csv")
# df1_paper_ids = set(df1['paper_id'])
# df2_paper_ids = set(df2['paper_id'])
# all_annotated_paper_ids = df1_paper_ids | df2_paper_ids

# for index, row in df.iterrows():
#     # progress update
#     if index % 100 == 0: print('...processing row %d' % (index))

#     # if this paper_id doesn't already have an annotation
#     if row['paper_id'] not in all_annotated_paper_ids:
#         # check that row is not na
#         if pd.notna(row['text']):
#             doc = nlp(row['text'])

#             sents = list(doc.sents)
#             # iterate over sentences
#             for s in range(1, len(sents)-1):

#                 num_sents += 1
#                 count = 0

#                 previous_sent =  sents[s-1].text.lower()
#                 current_sent = sents[s].text.lower()
#                 next_sent = sents[s+1].text.lower()

#                 # if the context sentences are too short to be good, just continue
#                 if (len(previous_sent) < 5) or (len(next_sent) < 5):
#                     continue
                
#                 else:
#                     # get spacy noun chunks
#                     curr_sent_concepts = set(getPhrases(nlp(current_sent)))

#                     # get intersection of sentence concepts and all concepts
#                     curr_intersection = concept_set.intersection(curr_sent_concepts)

#                     concat = previous_sent + " " + current_sent + " " + next_sent
#                     if len(curr_intersection) > 0:
#                         paper_ids.append(row['paper_id'])
#                         sentences.append(concat)                
#                         concepts.append(curr_intersection)                    

#         # if na skip over this row
#         else: continue

# print("%d sentences in %d rows" % (num_sents, index))

# # make final output df
# df_output = pd.DataFrame(list(zip(paper_ids, sentences, concepts)), 
# columns =['paper_id', 'sentence', 'forecite_concepts']) 
# print(df_output)
# df_output.to_csv("./score-threshold-1-0/3-sentence-min1-noun-phrase.csv")

#----------------------------------------------------------
# 2 continguous sentences: noun phrase level and substring level
#----------------------------------------------------------

# sentence1_concepts = []
# sentence2_concepts = []
# for index, row in df.iterrows():
#     # progress update
#     if index % 100 == 0:
#         print('...processing row %d' % (index))

#     # check that row is not na
#     if pd.notna(row['text']):
#         doc = nlp(row['text'])

#         sents = list(doc.sents)
#         # iterate over sentences
#         for s in range(len(sents)-1):
#             num_sents += 1
#             count = 0
#             current_sent = sents[s].text.lower()
#             next_sent = sents[s+1].text.lower()

#             current_sent_cased = sents[s].text
#             next_sent_cased = sents[s+1].text

#             # Option 1: get spacy noun chunks
#             curr_sent_concepts = set(getPhrases(nlp(current_sent)))
#             next_sent_concepts = set(getPhrases(nlp(next_sent)))

#             # Option 2: get word-level substrings for spacy noun chunks
#             # curr_sent_concepts = set(getPhraseSubstrings(nlp(current_sent)))
#             # next_sent_concepts = set(getPhraseSubstrings(nlp(next_sent)))

#             # get intersection of sentence concepts and all concepts
#             curr_intersection = concept_set.intersection(curr_sent_concepts)
#             next_intersection = concept_set.intersection(next_sent_concepts)

#             # if row['paper_id'] == 12205351:
#             #     print(curr_intersection)
#             #     print(next_intersection)

#             concat = current_sent_cased + " " + next_sent_cased
#             if len(curr_intersection) > 0 and len(next_intersection) > 0:

#                 paper_ids.append(row['paper_id'])
#                 sentences.append(concat)                
#                 sentence1_concepts.append(curr_intersection)
#                 sentence2_concepts.append(next_intersection)
                

#     # if na skip over this row
#     else: continue

# print("%d sentences in %d rows" % (num_sents, index))

# # make final output df
# df_output = pd.DataFrame(list(zip(paper_ids, sentences, sentence1_concepts, sentence2_concepts)), 
# columns =['paper_id', 'sentences', 'sentence1_concepts', 'sentence2_concepts']) 
# print(df_output)
# df_output.to_csv("./score-threshold-1-0/2-contiguous-sentences-noun-phrases-only-originalcase.csv")


#----------------------------------------------------------
# single sentence: noun phrase level and substring level
#----------------------------------------------------------

# count = 0
# num_1concept = 0
# for index, row in df.iterrows():
#     # progress update
#     if index % 100 == 0:
#         print('...processing row %d' % (index))

#     # check that row is not na
#     if pd.notna(row['text']):
#         doc = nlp(row['text'])
#         # iterate over sentences
#         for sent in doc.sents:

#             num_sents += 1
#             # count = 0
#             text = sent.text.lower()
            
#             # get noun phrase level concepts
#             phrase_keys = getPhrases(nlp(text))

#             # # get all word-level substrings concepts
#             # phrase_keys = getPhraseSubstrings(nlp(text))

#             # make set out of phrases for this sentence
#             sentence_concept_set = set(phrase_keys)

#             # get intersection of sentence concepts and all concepts
#             intersection = concept_set.intersection(sentence_concept_set)

#             # get all single sentences with 1+ noun phrase
#             if len(intersection) > 0:
#                 count+=1
#                 if len(intersection) == 1:
#                     num_1concept += 1
#                 paper_ids.append(row['paper_id'])
#                 concepts.append(intersection)
#                 sentences.append(sent.text)

#     # if na skip over this row
#     else: continue


# print(count)
# print(num_1concept)
# print("%d sentences in %d rows" % (num_sents, index))

# # make final output df
# df_output = pd.DataFrame(list(zip(paper_ids, sentences, concepts)), 
# columns =['paper_id', 'sentence', 'forecite_concepts']) 
# print(df_output)

# df_output.to_csv("./score-threshold-1-0/single-sentence-min1-noun-phrase-originalcase.csv")
