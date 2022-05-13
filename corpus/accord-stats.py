import pandas as pd
import numpy as np

#----------------------------------------------------------
# get stats for ACL Demo track submission
#----------------------------------------------------------

df1 = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/ACCoRD/1-sentence-annotations copy.csv")
df2 = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/ACCoRD/2-sentence-annotations copy.csv")
df_all = pd.concat([df1, df2])

# save df with only positive rows
rslt_df = df_all[df_all['is_relational']==1]

#----------------------------------------------------------
print("\n----BASIC STATS-----")

# get number of positive and negative examples in ACCoRD
print("Number of rows = %d" % len(df_all))
print("Number of extractions with a positive label = %d" % len(df_all[df_all['is_relational']==1]))
print("Number of extractions with a negative label = %d" % len(df_all[df_all['is_relational']==0]))
print("Number of extractions with is_relational==nan = %d" % len(df_all[pd.isna(df_all['is_relational'])]))
print("Number of formatted statements = %d" % len(df_all[pd.notna(df_all['formatted_statement'])]))
print("--> Number of formatted statements made from an extraction = %d" % len(df_all[pd.notna(df_all['formatted_statement']) & (df_all['is_relational']==1)]))
print("--> Number of additional formatted statements made from an extraction = %d" % len(df_all[pd.notna(df_all['formatted_statement']) & pd.isna(df_all['is_relational'])]))


# get number of instances without a differentia
num_without_differentia = 0
for index, row in rslt_df.iterrows():
    if pd.notna(row['formatted_statement']):
        if row['formatted_statement'].endswith(row['concept_b']):
            num_without_differentia+=1
print("Number of formatted sentences with no differentia (end with the annotated concept B) = %d" % num_without_differentia)

#----------------------------------------------------------
print("\n----DESCRIPTION + CONCEPT B STATS-----")
group_count_conceptb = rslt_df.groupby('concept_a')['concept_b'].nunique().reset_index(name='num_unique_concept_bs')
group_count_descriptions = rslt_df.groupby(by=["concept_a"]).size().reset_index(name='counts')
# merge dfs on concept_a column
merged = pd.merge(group_count_conceptb, group_count_descriptions, on="concept_a")
print(np.average(merged['counts']))
print(merged['counts']. value_counts())
merged.to_csv("test.csv")


#----------------------------------------------------------
print("\n----ERRORS-----")
# ERRORS
print("rows that don't have a formatted statement but are also not marked 0 for is_relational\n")
print(df_all[pd.isna(df_all['formatted_statement']) & pd.isna(df_all['is_relational'])])
print(df_all[pd.isna(df_all['formatted_statement']) & (df_all['is_relational']==1)])