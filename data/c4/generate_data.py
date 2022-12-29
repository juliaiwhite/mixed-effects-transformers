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

short_names = ['https://www.frontier','https://www.chicagot','https://link.springe',
               'https://www.aljazeer','https://www.instruct','https://www.npr.org/',
               'https://www.dailymai','https://www.csmonito','https://www.baltimor', 
               'http://www.city-data']

site_files = {'frontiersin.org':open('files/temp_frontiersin.org_data.json','w'),
              'chicagotribune.com':open('files/temp_chicagotribune.com_data.json','w'),
              'link.springer.com':open('files/temp_link.springer.com_data.json','w'),
              'aljazeera.com':open('files/temp_aljazeera.com_data.json','w'),
              'instructables.com':open('files/temp_instructables.com_data.json','w'),
              'npr.org':open('files/temp_npr.org_data.json','w'),
              'dailymail.co.uk':open('files/temp_dailymail.co.uk_data.json','w'),
              'csmonitor.com':open('files/temp_csmonitor.com_data.json','w'),
              'baltimoresun.com':open('files/temp_baltimoresun.com_data.json','w'),
              'city-data.com':open('files/temp_city-data.com_data.json','w')}
url_values = {}
url_idx = 1
site_values = {site:i+1 for i,site in enumerate(site_files.keys())}
i = {site:0 for site in site_values.keys()}

done_looping = False
while not done_looping:
    try:
        row = next(iterator)
        if row['url'][:20] in short_names:
            site = list(site_values.keys())[short_names.index(row['url'][:20])]
            if site in row['url']:
                site_id = site_values[site]
                text = row['text']

                url = row['url']
                if url in url_values:
                    url_id = url_values[url]
                else:
                    url_id = url_idx
                    url_values[url] = url_idx
                    url_idx += 1

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
                json.dump(url_values, open("url_values.json",'w'))
    except StopIteration:
        done_looping = True
            
for site, file in site_files.items():
    file.close()

for site in site_values.keys():
    os.system('../terashuf/terashuf < files/temp_'+site+'_data.json > files/'+site+'_data.json')
    os.system('rm files/temp_'+site+'_data.json')
    
print(i)
json.dump(i, open("data_info.json",'w'))
