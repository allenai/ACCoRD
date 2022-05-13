import os
import json
import pandas as pd
import pickle

concept_dict = {}
# for files in the data directory
count = 1
for filename in os.listdir("/net/nfs2.s2-research/soniam/concept-rel/resources/forecite/noun-phrase-scores-top-papers/"):
    print("getting concepts for file %d" % count)
    if filename.endswith(".json"):
        # open the json file
        with open("/net/nfs2.s2-research/soniam/concept-rel/resources/forecite/noun-phrase-scores-top-papers/%s" % filename) as f:
            # iterate over lines in this file
            for line in f:
                data = json.loads(line)
                concept = data['phrase']
                score = data['score']
                n = data['n']
                paper_id = data['corpus_paper_id']
                concept_dict[concept] = (score, n, paper_id)
    
    count += 1

with open('./forecite_concept_score_count_dict.pickle', 'wb') as handle:
    pickle.dump(concept_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("pickled concept dictionary")

# with open('concept-scores-counts.json', 'w') as fp:
#     json.dump(concept_dict, fp)