# Script for processing COCO captions
# 
# Tokenizes text
# Adds <START> and <END> tokens
# Replaces uncommon words with special <UNK> token
# Pads captions with <NULL> token

import argparse
import json
import boto3
import sagemaker
import spacy

from typing import Optional

MIN_COUNT = 4
CAP_LENGTH = 17

class Token():
    # Class used to analyze number of occurences per token in vocab
    def __init__(self, text:str, count:Optional[int] = 1):
        self.text = text
        self.count = count
        
    def inc(self):
        self.count += 1
        
    def __iter__(self):
        return iter((self.count, self.text))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', help='the suffix of the caption file to process')
    args = parser.parse_args()
    
    bucket = sagemaker.Session().default_bucket()
    bucket = boto3.resource('s3').Bucket(bucket)
    
    path = f'annotations/captions_{args.split}.json'
    dataset = json.load(bucket.Object(path).get()['Body'])
    num_anns = len(dataset['annotations'])

    nlp = spacy.load('en_core_web_md')

    vocab = {0:Token('<NULL>', MIN_COUNT), 1:Token('<START>', MIN_COUNT), 2:Token('<END>', MIN_COUNT), 3:Token('<UNK>', MIN_COUNT)}
    spacy_2_emb = {} # maps spacy embedding row # to torch embedding row #
    for i in range(num_anns):
        caption = dataset['annotations'][i]['caption']
        doc = nlp(caption)

        ids = [1]
        for token in doc:
            if token.is_punct:
                continue # ignore punctuation

            if nlp.vocab.has_vector(token.norm):
                id = nlp.vocab.vectors.key2row[token.norm] # encode tokens using index in vector matrix

                # keep track of number of token occurrences
                if id in spacy_2_emb:
                    vocab[spacy_2_emb[id]].inc()
                else:
                    spacy_2_emb[id] = len(vocab)
                    vocab[spacy_2_emb[id]] = Token(token.text)
                    
                id = spacy_2_emb[id]
            else:
                id = 3

            ids.append(id)
        ids.append(2)
        
        dataset['annotations'][i]['target'] = ids

    # distill vocab
    vocab = {k:v.text for k, v in vocab.items() if v.count >= MIN_COUNT}

    # replace rare tokens with <UNK> and pad with <NULL>
    for i in range(num_anns):
        tokens = dataset['annotations'][i]['target']
        if len(tokens) <= CAP_LENGTH:
            tokens = [tok if tok in vocab else 3 for tok in tokens]
            tokens += [0] * (CAP_LENGTH - len(tokens))
            dataset['annotations'][i]['target'] = tokens
        else:
            dataset['annotations'][i]['target'] = None # ignore long captions

    dataset = json.dumps(dataset).encode('utf-8')
    vocab = json.dumps(vocab).encode('utf-8')
    
    bucket.put_object(Body=dataset, Key=f'annotations/targets_{args.split}.json')
    bucket.put_object(Body=vocab, Key=f'annotations/vocab_{args.split}.json')
        
if __name__ == '__main__':
    main()