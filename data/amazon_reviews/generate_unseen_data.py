import datasets
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

download_config = datasets.utils.DownloadConfig(cache_dir='/mnt/fs2/juliawhi/.huggingface/downloads/')
categories = {"sports":"Sports_v1_00"}

product_values = json.load(open("product_values.json",'r'))
customer_values = json.load(open("customer_values.json",'r'))
category_values = {'sports':0}
i = {category:0 for category in category_values.keys()}
for category in category_values.keys():
    category_id = category_values[category]
    corpus = datasets.load_dataset('amazon_us_reviews', categories[category], download_config=download_config).map(None)
    file = open('files/temp_unseen_data.json','w')
    for utt in tqdm(corpus['train']):
        text = utt['review_body']

        product = utt['product_id']
        if product in product_values:
            product_id = product_values[product]
        else:
            product_id = 0

        customer = utt['customer_id']
        if customer in customer_values:
            customer_id = customer_values[customer]
        else:
            customer_id = 0

        sentences = nltk.tokenize.sent_tokenize(text)
        for sentence in sentences:
            sentence = tokenizer(sentence)['input_ids']
            if len(sentence)<=1024:
                line = "{\"sentence\":" + str(sentence) + "," + \
                       "\"category\":\"" + category + "\"," + \
                       "\"category_id\":" + str(category_id) + "," + \
                       "\"product\":\"" + product + "\"," + \
                       "\"product_id\":" + str(product_id) + "," + \
                       "\"customer\":\"" + customer + "\"," + \
                       "\"customer_id\":" + str(customer_id) + "}"
                if i[category] != 0:
                    line = '\n'+line
                file.write(line)
                i[category] += 1
    file.close()
    
for category in category_values.keys():
    os.system('../terashuf/terashuf < files/temp_unseen_data.json > files/unseen_data.json')
    os.system('rm files/temp_unseen_data.json')
    
print(i)
