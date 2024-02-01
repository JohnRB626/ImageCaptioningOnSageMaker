# Script for processing COCO captions
# 
# Creates vocab
#     Indexes tokens
# 
# Tokenizes text
#     Converts strings to lists of token ids
#     Adds <START> and <END> tokens
#     Replaces uncommon words with special <UNK> token
#     Pads captions with <NULL> token

import argparse
import json
import boto3
import sagemaker
import spacy

from typing import Optional

MIN_COUNT = 4
CAP_LENGTH = 15

class Token():
    # Class used to count number of occurences per token in vocab
    def __init__(self, text:str, count:Optional[int] = 1):
        self.text = text
        self.count = count
        
    def inc(self):
        self.count += 1
        
    def __iter__(self):
        return iter((self.count, self.text))

def create_vocab(split:str):
    """
    Creates a vocab and tokenizes captions using spacy vector indices.
    
    Vocab is only created for train splits
    """
    
    bucket = sagemaker.Session().default_bucket()
    bucket = boto3.resource('s3').Bucket(bucket)
    
    path = f'annotations/captions_{split}.json'
    dataset = json.load(bucket.Object(path).get()['Body'])

    nlp = spacy.load('en_core_web_md')
    
    vocab = {}
    num_anns = len(dataset['annotations'])
    for i in range(num_anns):
        caption = dataset['annotations'][i]['caption']
        doc = nlp(caption)

        ids = []
        for token in doc:
            if token.is_punct:
                continue # ignore punctuation

            if nlp.vocab.has_vector(token.norm):
                id = nlp.vocab.vectors.key2row[token.norm] # encode tokens using index in spacy vector matrix

                # keep track of number of token occurrences
                if id in vocab:
                    vocab[id].inc()
                else:
                    vocab[id] = Token(token.text)
            else:
                id = -1 # unknown token

            ids.append(id)
        
        dataset['annotations'][i]['target'] = ids
        
    bucket.put_object(
        Body=json.dumps(dataset).encode('utf-8'),
        Key=f'annotations/targets_{split}.json'
    )
    
    #save vocab objects for use during retokenization
    if 'train' in split:
        spacy_vocab = {k:v.text for k, v in vocab.items() if v.count >= MIN_COUNT}
        vocab = {e:v for e, v in enumerate(spacy_vocab.values(), 4)} # start at 4 to leave room for <META> tokens
        spacy2model = {old:new for old, new in zip(spacy_vocab.keys(), vocab.keys())}
        
        vocab[0] = '<NULL>'
        vocab[1] = '<START>'
        vocab[2] = '<END>'
        vocab[3] = '<UNK>'
        
        bucket.put_object(
            Body=json.dumps(vocab).encode('utf-8'),
            Key='annotations/vocab.json'
        )
        bucket.put_object(
            Body=json.dumps(spacy2model).encode('utf-8'),
            Key='annotations/spacy2model.json'
        )

        
def retokenize(split:str):
    """
    Retokenizes captions via a vocab reindexed with consecutive integers beginning at 0.
    
    Additionally, adds <START>, <END>, <UNK>, and <NULL> tokens
    """
    
    bucket = sagemaker.Session().default_bucket()
    bucket = boto3.resource('s3').Bucket(bucket)
    
    dataset = json.load(bucket.Object(f'annotations/targets_{split}.json').get()['Body'])    
    vocab = json.load(bucket.Object('annotations/vocab.json').get()['Body'])
    index_map = json.load(bucket.Object('annotations/spacy2model.json').get()['Body'])

    num_anns = len(dataset['annotations'])
    for i in range(num_anns):
        tokens = dataset['annotations'][i]['target']
        if len(tokens) <= CAP_LENGTH:
            tokens = [index_map[str(tok)] if str(tok) in index_map else 3 for tok in tokens]
            tokens = [1] + tokens + [2]
            tokens += [0] * (CAP_LENGTH + 2 - len(tokens)) # add 2 for <START> and <END> tokens
            dataset['annotations'][i]['target'] = tokens
        else:
            dataset['annotations'][i]['target'] = None # ignore long captions

    dataset = json.dumps(dataset).encode('utf-8')
    bucket.put_object(Body=dataset, Key=f'annotations/targets_{split}.json')
    
    
def full(split:str):
    create_vocab(split)
    retokenize(split)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        'function', type=str, choices=['create_vocab', 'retokenize', 'full'], help='function to call'
    )
    parser.add_argument(
        'split', type=str, help='the suffix of the caption file to process'
    )
    
    args = parser.parse_args()
    
    globals()[args.function](args.split)