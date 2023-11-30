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

class Token():
    # Class used to analyze number of occurences per token in vocab
    def __init__(self, text:str):
        self.text = text
        self.count = 1
        
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

    vocab = {}
    for i in range(num_anns):
        caption = dataset['annotations'][i]['caption']
        doc = nlp(caption)

        ids = [20000] # <START> token
        for token in doc:
            if token.is_punct:
                pass # ignore punctuation

            elif nlp.vocab.has_vector(token.norm):
                id = nlp.vocab.vectors.key2row[token.norm] # encode tokens using index in vector matrix

                # keep track of number of token occurrences
                if id in vocab:
                    vocab[id].inc()
                else:
                    vocab[id] = Token(token.text)
            else:
                id = 20002 # <UNK> token

            ids.append(id)

        ids.append(20001) # <END> token
        dataset['annotations'][i]['target'] = ids

    # finalize vocab
    vocab = {k:v.text for k, v in vocab.items() if v.count > 3}

    # replace rare tokens with <UNK> and pad with <NULL>
    cap_length = 17
    for i in range(num_anns):
        tokens = dataset['annotations'][i]['target']
        if len(tokens) <= cap_length:
            tokens = [tok if tok in vocab else 20002 for tok in enc_cap]
            tokens += [20003] * (cap_length - len(enc_cap))
            dataset['annotations'][i]['target'] = tokens
        else:
            dataset['annotations'][i]['target'] = None # ignore long captions

    dataset = json.dumps(dataset).encode('utf-8')
    vocab = json.dumps(vocab).encode('utf-8')
    
    bucket.put_object(Body=dataset, Key=f'annotations/targets_{args.split}.json')
    bucket.put_object(Body=vocab, Key=f'annotations/vocab_{args.split}.json')
        
if __name__ == '__main__':
    main()