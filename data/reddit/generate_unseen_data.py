from convokit import Corpus, download
from transformers import GPT2Tokenizer
import nltk

from tqdm import tqdm
import json
import os

import math
import random
random.seed(123)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

file_path = os.path.expanduser("~/.convokit/downloads/")

speaker_values = json.load(open("speaker_values.json",'r'))
subreddit_values = {'AskReddit':0}
i = {subreddit:0 for subreddit in subreddit_values.keys()}
for subreddit in subreddit_values.keys():
    subreddit_id = subreddit_values[subreddit]
    print('processing subreddit '+subreddit+'...')
    data_file = open(file_path+"subreddit-"+subreddit+"/utterances.jsonl","r")
    file = open('files/temp_unseen_data.json','w')
    for utt in tqdm(data_file):
        utt = json.loads(utt)
        text = utt['text']
        if text not in ['[deleted]','[removed]']:

            score = utt['meta']['score']
            score_id = 2 if score>0 else 1

            speaker = utt['user']
            if speaker in speaker_values:
                speaker_id = speaker_values[speaker]
            else:
                speaker_id = 0

            sentences = nltk.tokenize.sent_tokenize(text)
            for sentence in sentences:
                sentence = tokenizer(sentence)['input_ids']
                if len(sentence)<=1024:
                    line = "{\"sentence\":" + str(sentence) + "," + \
                           "\"subreddit\":\"" + subreddit + "\"," + \
                           "\"subreddit_id\":" + str(subreddit_id) + "," + \
                           "\"score\":" + str(score) + "," + \
                           "\"score_id\":" + str(score_id) + "," + \
                           "\"speaker\":\"" + speaker + "\"," \
                           "\"speaker_id\":" + str(speaker_id) + "}"
                    if i[subreddit] != 0:
                        line = '\n'+line
                    file.write(line)
                    i[subreddit] += 1
    file.close()

for subreddit in subreddit_values.keys():
    os.system('../terashuf/terashuf < files/temp_unseen_data.json > files/unseen_data.json')
    os.system('rm files/temp_unseen_data.json')
    
print(i)
