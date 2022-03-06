import argparse
import codecs
import csv
import os
import re
import sys

from transformers import RobertaTokenizer
import numpy as np

class DataPrepper:
    '''
    An alternative to using process_data.py. The advantage
    is that this version can be used from the command line,
    like the original. But this class can also be imported
    and used in a workflow.
    
    An instance of this class takes a raw input file, such
    as sample_input_ER.csv, emotional-reactions-reddit.csv,
    or interpretations-reddit.csv.
    
    The instance tokenizes and otherwise process the input
    file to create a corresponding file that is suitable 
    for training a model.
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, input_csv_file, output_csv_file):
        
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', 
                                                     do_lower_case=True)

        input_file = codecs.open(input_csv_file, 'r', 'utf-8')
        output_file = codecs.open(output_csv_file, 'w', 'utf-8')
        
        csv_reader = csv.reader(input_file, delimiter = ',', quotechar='"')
        csv_writer = csv.writer(output_file, delimiter = ',',quotechar='"') 
        
        next(csv_reader, None) # skip the header
        
        csv_writer.writerow(["id","seeker_post","response_post","level","rationale_labels","rationale_labels_trimmed","response_post_masked"])
        
        for row in csv_reader:
            # sp_id,rp_id,seeker_post,response_post,level,rationales
        
            seeker_post = row[2].strip()
            response = row[3].strip()
        
            response_masked = response
        
            response_tokenized = tokenizer.decode(tokenizer.encode_plus(
                response, 
                add_special_tokens=True, 
                max_length = 64,
                truncation=True,
                padding='max_length')['input_ids'], clean_up_tokenization_spaces=False)
        
            response_tokenized_non_padded = tokenizer.decode(tokenizer.encode_plus(
                response, 
                add_special_tokens=True, 
                max_length=64,
                truncation=True,
                padding='max_length')['input_ids'], clean_up_tokenization_spaces=False)
        
            response_words = tokenizer.tokenize(response_tokenized)
            response_non_padded_words = tokenizer.tokenize(response_tokenized_non_padded)
        
            if len(response_words) != 64:
                continue
        
            response_words_position = np.zeros((len(response),), dtype=np.int32)
        
            rationales = row[5].strip().split('|')
        
            rationale_labels = np.zeros((len(response_words),), dtype=np.int32)
        
        
            curr_position = 0
        
            for idx in range(len(response_words)):
                curr_word = response_words[idx]
                if curr_word.startswith('Ġ'):
                    curr_word = curr_word[1:]
                response_words_position[curr_position: curr_position+len(curr_word)+1] = idx
                curr_position += len(curr_word)+1
        
            if len(rationales) == 0 or row[5].strip() == '':
                rationale_labels[1:len(response_non_padded_words)] = 1
                response_masked = ''
        
            for r in rationales:
                if r == '':
                    continue
                try:
                    r_tokenizer = tokenizer.decode(tokenizer.encode(r, add_special_tokens = False))
                    match = re.search(r_tokenizer , response_tokenized)
        
                    curr_match = response_words_position[match.start(0):match.start(0)+len(r_tokenizer)]
                    curr_match = list(set(curr_match))
                    curr_match.sort()
        
                    response_masked = response_masked.replace(r, ' ')
                    response_masked = re.sub(r' +', ' ', response_masked)
        
                    rationale_labels[curr_match] = 1
                except:
                    continue
            
            
            rationale_labels_str = ','.join(str(x) for x in rationale_labels)
        
            rationale_labels_str_trimmed = ','.join(str(x) for x in rationale_labels[1:len(response_non_padded_words)])
        
        
            csv_writer.writerow([row[0] + '_' + row[1], seeker_post, response, row[4], rationale_labels_str, len(rationale_labels_str_trimmed), response_masked])
        input_file.close()
        output_file.close()

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("process_data")
    parser.add_argument("--input_path", type=str, help="path to input data")
    parser.add_argument("--output_path", type=str, help="path to output data")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Input {args.input_path} does not exist.")
        sys.exit(1)

    if os.path.exists(args.output_path):
        answer = input(f"Outfile {args.output_path} exists; overwrite? (y/n)")
        if answer not in ['y', 'Y', 'yes', 'Yes']:
            print("Aborting operation")
            sys.exit(1)
    
    DataPrepper(args.input_path, args.output_path)
