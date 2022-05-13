import os
import json
import pandas as pd
import csv   

# USE THIS SCRIPT ON SERVER 1 TO ACCESS LOCAL COPY OF S2ORC

# filter papers using metadata values
def getValidPaperIDs(batch_id):
    all_valid_ids = []
    count = 0
    with open(f'/disk2/s2orc/20200705v1/full/metadata//metadata_{batch_id}.jsonl') as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']

            # suppose we only care about CS papers and papers with pdf parses available
            if (metadata_dict['mag_field_of_study'] == None) or ('Computer Science' not in metadata_dict['mag_field_of_study']):
                continue
                
            if not metadata_dict['has_pdf_parse']:
                continue
                
            if not metadata_dict['has_pdf_parsed_body_text']:
                continue
                
            all_valid_ids.append(paper_id)
            count +=1
                
    print("number of valid paper IDs = % d" % count)
    return all_valid_ids

# copy full parses of papers that meet our criteria
def copyValidPaperIDs(all_valid_ids, batch_id):
    count = 0
    for paper_id in all_valid_ids:
        count += 1
        directory = paper_id[:len(paper_id)-4]
        file_name = paper_id[-4:]

        if not os.path.exists("./metadata-%s/%s.json" % (batch_id, paper_id)):
            os.system("cp /disk2/s2orc/20200705v1/expanded/%s/%s.json ./metadata-%s/%s.json" % (directory, file_name, batch_id, paper_id))

        if count % 100 == 0:
            print("copied %d papers from metadata %d" % (count, batch_id))


# get desired sections of full pdf parses
def getPaperSections(batch_id):
    all_text = []
    for filename in os.listdir("./metadata-%d/" % batch_id):
        path = "./metadata-%d/%s" % (batch_id, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path) as f_pdf:
                # dict of metadata keys
                pdf_parse_dict = json.loads(f_pdf.read())
                subject = pdf_parse_dict['mag_field_of_study']
                paper_id = pdf_parse_dict['paper_id']
                
                # dict of pdf parse related keys
                pdf_parse_dict = pdf_parse_dict['pdf_parse']

                # (2) pull out fields we need from the pdf parse, including abstract & text
                paragraphs = pdf_parse_dict['abstract'] + pdf_parse_dict['body_text']

                # (3) loop over paragraphs
                for paragraph in paragraphs:
                    section_heading = paragraph['section'].lower()
                    sections_of_interest = []
                    # sections we care about:
                    AIR = ['abstract', 'introduction', 'previous', 'related work', 'related literature', 'background', 'motivation']
                    for i in AIR:
                        if i in section_heading:
                            sections_of_interest.append(paragraph)  

                    # (4) loop over each section in this paragraph
                    for section in sections_of_interest:
                        all_text.append({
                            'paper_id': paper_id,
                            'subject': subject,
                            'text': paragraph['text'].encode('ascii', 'ignore').decode('ascii')
                        })

    return all_text

def processText(all_text, output_file):
    df = pd.DataFrame.from_dict(all_text)
    print("total dataset = %s entries" % len(df))
    print("number of unique papers = %d" % len(pd.unique(df['paper_id'])))
    print("removing missing values...")
    df = df.dropna() # remove missing values
    print("total dataset = %s entries" % len(df))
    print("number of unique papers = %d" % len(pd.unique(df['paper_id'])))
    print("removing duplicates...")
    df = df.drop_duplicates(subset=['text'])
    print("total dataset = %s entries" % len(df))
    print("number of unique papers = %d" % len(pd.unique(df['paper_id'])))

    df.to_csv(output_file, index=False)


# main
for batch_id in range(30, 31):
    # make a directory for the papers from this batch_id
    os.system("mkdir metadata-%d" % batch_id)
    all_valid_ids = getValidPaperIDs(batch_id)
    copyValidPaperIDs(all_valid_ids, batch_id)
    all_text = getPaperSections(batch_id)
    processText(all_text, "./text-batch-id-%d.csv" % batch_id)
