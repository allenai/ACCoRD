from os import rename
import pandas as pd
import re

def renameDfColumns(version):
    df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/user-study/demo-evaluation/study-responses-%s.csv" % version)

    # iterate over remaining columns and rename
    for label, content in df.iteritems():
        if label == "Timestamp":
            df.rename(columns={label:"timestamp"},inplace=True)
        if label == "Email Address":
            df.rename(columns={label:"email"},inplace=True)
        if label == "I consider myself proficient in the following areas of computer science:":
            df.rename(columns={label:"proficiency_areas"},inplace=True)
        if label == "How many years of experience do you have in NLP?":
            df.rename(columns={label:"years_experience"},inplace=True)
        if label == "Have you published a peer-reviewed academic paper in NLP?":
            df.rename(columns={label:"has_published"},inplace=True)
        if label == "What is the highest level of education you've completed (in computer science)?":
            df.rename(columns={label:"highest_edu"},inplace=True)

        # handle label names with concepts 
        if "[" in label:
            result = re.findall('\[.*?\]',label)
            if len(result) == 1:
                concept = result[0].lower().strip("[]").split()
                concept = "".join(concept)
            elif len(result) == 2:
                concept = result[0].lower().strip("[]").split()
                concept = "".join(concept)

                question = result[1].strip("[]")

            # append question type
            if "How well do you know this concept?" in label:
                question = "expertise"
            if "Imagine that you’re reading a paper" in label:
                question = "set_preference"

            final_label = concept + "_" + question

            df.rename(columns={label:final_label},inplace=True)


        # handle free response question
        if "Please describe how you evaluated/determined your preference for the above sets of descriptions." in label:
            df.rename(columns={label:"set_description_free_text"},inplace=True)
        if "Can you explain to me why you preferred certain descriptions over others?" in label:
            df.rename(columns={label:"individual_description_free_text"},inplace=True)
        if "Here, we asked you to rate each description individually." in label:
            df.rename(columns={label:"indivudal_vs_set_free_text"},inplace=True)
        
    print(df.keys())

    return df    


def mapResponses(df, version):
    if version == "versionA":
        df = df.replace('Set A', 'extractions_our_ranking')
        df = df.replace('Set B', 'generations_intuitive_ranking')
        df = df.replace('Set C', 'generations_our_ranking')

    elif version == "versionB":
        df = df.replace('Set A', 'generations_our_ranking')
        df = df.replace('Set B', 'extractions_our_ranking')
        df = df.replace('Set C', 'generations_intuitive_ranking')

    elif version == "versionC":
        df = df.replace('Set A', 'generations_intuitive_ranking')
        df = df.replace('Set B', 'generations_our_ranking')
        df = df.replace('Set C', 'extractions_our_ranking')

    # add column for version to each df
    df['version'] = [version] * len(df)

    # map description preference responses
    df = df.replace("👎 I would NOT want to see this description of the concept", "dont_want")
    df = df.replace("👍 I would want to see this description of the concept", "want")
    df = df.replace("⚫ No preference/opinion", "no_preference")
    
    return df

dfA = renameDfColumns('versionA')
dfB = renameDfColumns('versionB')
dfC = renameDfColumns('versionC')

dfA = mapResponses(dfA, 'versionA')
dfB = mapResponses(dfB, 'versionB')
dfC = mapResponses(dfC, 'versionC')

df_all = pd.concat([dfA, dfB, dfC]).reset_index()
print(df_all)
df_all.to_csv("/net/nfs2.s2-research/soniam/concept-rel/user-study/demo-evaluation/study-responses-all.csv")