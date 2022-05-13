from re import A
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats import inter_rater as irr
import numpy as np
from statistics import mean

df_expertise_ratings = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/user-study/demo-evaluation/expertise-ratings.csv")
df_median_expertise = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/user-study/demo-evaluation/median-expertise-ratings.csv")
df_description_preferences = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/user-study/demo-evaluation/description-preferences.csv")

df_error_analysis = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/user-study/demo-evaluation/completed-error-analysis.csv")
error_analysis_concepts = set(df_error_analysis['forecite_concept'].to_list())

# remove _expertise to isolate concept
concepts = list(map(lambda x: x.replace('_expertise',''), df_median_expertise['concept_expertise']))
df_expertise_ratings['concept_expertise'] = list(map(lambda x: x.replace('_expertise',''), df_expertise_ratings['concept_expertise']))
df_median_expertise['concept_expertise'] = list(map(lambda x: x.replace('_expertise',''), df_median_expertise['concept_expertise']))

# rename concept columns
df_median_expertise = df_median_expertise.rename(columns={"concept_expertise": "concept"})
df_expertise_ratings = df_expertise_ratings.rename(columns={"concept_expertise": "concept"})

print(df_expertise_ratings.keys())
print(df_description_preferences.keys())
# merge expertise ratings and description preferences dfs
df_expertise_preferences = pd.merge(df_expertise_ratings, df_description_preferences, on=["concept", "email"])
print(df_expertise_preferences.keys())
res = df_expertise_preferences[['email', 'concept', 'expertise', 'description', 'preference_rating']]
res.to_csv("merged-expertise-description-preferences.csv")


#--------------------------------------------------------
# Do people with more expertise tend to agree with each 
# other more than they do with those with less expertise?
#--------------------------------------------------------

# function to select and format description preferences for each segment
# and calculate Fleiss' kappa
def calculateKappaPerSegment(df_concept_description_preferences, segmentEmails):
    # select rows of the description preferences df for this concept where emailIDs are in the segment list 
    df_concept_description_preferences_segment = df_concept_description_preferences.loc[df_concept_description_preferences['email'].isin(segmentEmails)]
    # select relevant columns from each df
    df_concept_description_preferences_segment = df_concept_description_preferences_segment[['email', 'description', 'preference_rating']]
    # convert categorical preference ratings to numeric
    df_concept_description_preferences_segment['preference_rating'] = pd.factorize(df_concept_description_preferences_segment['preference_rating'])[0]
    # pivot df to get columns = emails, rows = descriptions
    np_concept_description_preferences_segment = df_concept_description_preferences_segment.pivot(index='description', columns='email', values='preference_rating').to_numpy()
    # calculate kappa for segment
    kappa = irr.fleiss_kappa(irr.aggregate_raters(np_concept_description_preferences_segment)[0], method='fleiss')

    return kappa

# #--------------------------------------------------------
# # store each kappa calculation for all concepts
# above_median_concept_kappas = []
# below_median_concept_kappas = []
# full_concept_kappas = []
# count = 0
# n_runs = 5

# # for each concept
# for concept in concepts:
#     # select rows for this concept
#     df_concept_description_preferences = df_description_preferences[df_description_preferences['concept']==concept]
#     all_emails = list(set(df_concept_description_preferences['email'].to_list()))

#     # # get median expertise for this concept
#     # median = df_median_expertise.loc[df_median_expertise['concept_expertise'] == concept, 'median.expertise'].iloc[0]
#     # global median
#     median = 4
#     # # global average
#     # median = 3.35

#     # select rows with expertise rating > median
#     above_median_segment = df_expertise_ratings.loc[(df_expertise_ratings['concept'] == concept) & (df_expertise_ratings['expertise'] > median)]
#     # select rows with expertise rating < median
#     below_median_segment = df_expertise_ratings.loc[(df_expertise_ratings['concept'] == concept) & (df_expertise_ratings['expertise'] < median)]
    
#     # select rows with expertise rating < median
#     median_segment = df_expertise_ratings.loc[(df_expertise_ratings['concept'] == concept) & (df_expertise_ratings['expertise'] == median)]
    
#     above_median_kappas = []
#     below_median_kappas = []
#     for i in range(n_runs):
#         # randomly + equally assign participants with median expertise to above-median segment and below-median segment
#         part_50 = median_segment.sample(frac = 0.5, random_state=i)
#         # Creating dataframe with rest of the 50% values
#         rest_part_50 = median_segment.drop(part_50.index)

#         # concatenate portion of median segment to each above-median and below-median segment
#         # select emailID column
#         above_median_segment_emails = pd.concat([above_median_segment, part_50])['email'].to_list()
#         below_median_segment_emails = pd.concat([below_median_segment, rest_part_50])['email'].to_list()

#         print(len(above_median_segment_emails), len(below_median_segment_emails))
        

#         above_median_kappa = calculateKappaPerSegment(df_concept_description_preferences, above_median_segment_emails)
#         below_median_kappa = calculateKappaPerSegment(df_concept_description_preferences, below_median_segment_emails)
#         full_kappa = calculateKappaPerSegment(df_concept_description_preferences, all_emails)

#         above_median_kappas.append(above_median_kappa)
#         below_median_kappas.append(below_median_kappa)

#     # average all the kappas for this concept
#     avg_above_median_kappa = mean(above_median_kappas)
#     avg_below_median_kappa = mean(below_median_kappas)
    
#     # append to list for all concepts
#     above_median_concept_kappas.append(avg_above_median_kappa)
#     below_median_concept_kappas.append(avg_below_median_kappa)
#     full_concept_kappas.append(full_kappa)

#     print(concept, avg_below_median_kappa > full_kappa)
#     if (avg_above_median_kappa > full_kappa):
#         count+=1
#     # print(avg_above_median_kappa)
#     # print(avg_below_median_kappa)
#     # print(full_kappa)
    
# print(count)
# # format data for seaborn plot
# newlist = [x for x in above_median_concept_kappas if np.isnan(x) == False]
# print(sum(newlist)/len(newlist))
# print(sum(below_median_concept_kappas)/len(below_median_concept_kappas))
# print(sum(full_concept_kappas)/len(full_concept_kappas))

# kappas_long = above_median_concept_kappas + below_median_concept_kappas + full_concept_kappas
# conditions_long = np.repeat(['above', 'below', 'full'], 20)
# concepts_long = concepts + concepts + concepts

# # combine lists into a df to plot
# df_res_long = pd.DataFrame(list(zip(concepts_long, kappas_long, conditions_long)),
#               columns=['concept','kappa', 'condition'])


# df_res_long.to_csv("per-expertise-segment-kappas.csv")

# # # plot
# # sns.catplot(x="concept", y="kappa", hue="condition", kind="bar", data=df_res_long, palette="ch:.25")
# # plt.xticks(rotation='vertical')
# # plt.savefig("/net/nfs2.s2-research/soniam/concept-rel/user-study/demo-evaluation/expertise-segment-kappas-barplot-global-average.png", bbox_inches="tight")


#--------------------------------------------------------
# how often below-median users chose descriptions with error, compared to above-median ones?  
# This would help disentangle whether the agreement difference is due to novice users just making error, or something else.
n_runs = 5

for concept in error_analysis_concepts:
    # select rows for this concept
    df_concept_description_preferences = df_description_preferences[df_description_preferences['concept']==concept]
    all_emails = list(set(df_concept_description_preferences['email'].to_list()))

    print(df_concept_description_preferences)

    # global median
    median = 4

    # select rows with expertise rating > median
    above_median_segment = df_expertise_ratings.loc[(df_expertise_ratings['concept'] == concept) & (df_expertise_ratings['expertise'] > median)]
    # select rows with expertise rating < median
    below_median_segment = df_expertise_ratings.loc[(df_expertise_ratings['concept'] == concept) & (df_expertise_ratings['expertise'] < median)]
    
    # select rows with expertise rating < median
    median_segment = df_expertise_ratings.loc[(df_expertise_ratings['concept'] == concept) & (df_expertise_ratings['expertise'] == median)]
    
    above_median_kappas = []
    below_median_kappas = []
    for i in range(n_runs):
        # randomly + equally assign participants with median expertise to above-median segment and below-median segment
        part_50 = median_segment.sample(frac = 0.5, random_state=i)
        # Creating dataframe with rest of the 50% values
        rest_part_50 = median_segment.drop(part_50.index)

        # concatenate portion of median segment to each above-median and below-median segment
        # select emailID column
        above_median_segment_emails = pd.concat([above_median_segment, part_50])['email'].to_list()
        below_median_segment_emails = pd.concat([below_median_segment, rest_part_50])['email'].to_list()

        print(len(above_median_segment_emails), len(below_median_segment_emails))
        
