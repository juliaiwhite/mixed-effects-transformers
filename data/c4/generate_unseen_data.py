import datasets
from datasets import load_dataset
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

dataset = load_dataset("c4", "en", split='train', streaming=True)
iterator = iter(dataset)

site_files = {'journals.plos.org':open('files/temp_unseen_data.json','w')}
url_values = json.load(open("url_values.json",'r'))

site_values = {'journals.plos.org':0}
i = {site:0 for site in site_values.keys()}

done_looping = False
while not done_looping:
    try:
        row = next(iterator)
        site = 'journals.plos.org'
        if site in row['url']:
            site_id = site_values[site]
            text = row['text']

            url = row['url']

            if url in url_values:
                url_id = url_values[url]
            else:
                url_id = 0

            sentences = nltk.tokenize.sent_tokenize(text)
            for sentence in sentences:
                sentence = tokenizer(sentence)['input_ids']
                if len(sentence)<=1024:
                    line = "{\"sentence\":" + str(sentence) + "," + \
                           "\"site\":\"" + site + "\"," + \
                           "\"site_id\":" + str(site_id) + "," + \
                           "\"url\":\"" + url + "\"," + \
                           "\"url_id\":" + str(url_id) + "}"
                    if i[site] != 0:
                        line = '\n'+line
                    site_files[site].write(line)
                    i[site] += 1
    except StopIteration:
        done_looping = True
            
for site, file in site_files.items():
    file.close()

for site in site_values.keys():
    os.system('../terashuf/terashuf < files/temp_unseen_data.json > files/unseen_data.json')
    os.system('rm files/temp_unseen_data.json')
    
print(i)
json.dump(i, open("data_info.json",'w'))
