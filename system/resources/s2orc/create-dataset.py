import os
import json

# feel free to wrap this into a larger loop for batches 0~99
BATCH_ID = 0

# create a lookup for the pdf parse based on paper ID
paper_id_to_pdf_parse = {}
with open('./pdfparse.jsonl') as f_pdf:
    for line in f_pdf:
        pdf_parse_dict = json.loads(line)
        paper_id_to_pdf_parse[pdf_parse_dict['paper_id']] = pdf_parse_dict

# filter papers using metadata values
all_text = []
count = 0
with open('./metadata.jsonl') as f_meta:
    for line in f_meta:
        metadata_dict = json.loads(line)
        paper_id = metadata_dict['paper_id']
        # print("Currently viewing S2ORC paper: "+ paper_id)

        # suppose we only care about CS papers
        if (metadata_dict['mag_field_of_study'] == None) or ('Computer Science' not in metadata_dict['mag_field_of_study']):
            continue
            
        
        # get citation context (paragraphs)!
        if paper_id in paper_id_to_pdf_parse:
            print("Currently viewing S2ORC paper: "+ paper_id)
            print(metadata_dict['mag_field_of_study'])
            print("pdf parse is available")
            print("")
            # (1) get the full pdf parse from the previously computed lookup dict
            pdf_parse = paper_id_to_pdf_parse[paper_id]
            
            # (2) pull out fields we need from the pdf parse, including bibliography & text
            paragraphs = pdf_parse['abstract'] + pdf_parse['body_text']
            
            # (3) loop over paragraphs
            for paragraph in paragraphs:
                # sections we care about:
                AIR = ['abstract', 'Abstract', 'INTRODUCTION', 'Introduction', 'RELATED WORK', 'Related Work']
                sections_of_interest = []
                if paragraph['section'] in AIR:
                    sections_of_interest.append(paragraph)                
                
                # (4) loop over each section in this paragraph
                for section in sections_of_interest:
                    all_text.append({
                        'paper_id': paper_id,
                        'subject': metadata_dict['mag_field_of_study'],
                        'text': paragraph['text'].encode('ascii', 'ignore').decode('ascii')
                    })



import csv   

keys = all_text[0].keys()
with open('cleaned_text.csv', 'w')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_text)