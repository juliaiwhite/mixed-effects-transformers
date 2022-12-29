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
categories = {"video_games":"Video_Games_v1_00", "pet_products":"Pet_Products_v1_00", "grocery":"Grocery_v1_00", 
            "home":"Home_v1_00", "electronics":"Electronics_v1_00", "beauty":"Beauty_v1_00", "baby":"Baby_v1_00",
            "automotive":"Automotive_v1_00", "apparel":"Apparel_v1_00", "books":"Books_v1_00"}
product_values = {}
product_idx = 1
customer_values = {}
customer_idx = 1
category_values = {category:i+1 for i,category in enumerate(categories.keys())}
i = {category:0 for category in category_values.keys()}
for category in category_values.keys():
    category_id = category_values[category]
    corpus = datasets.load_dataset('amazon_us_reviews', categories[category], download_config=download_config).map(None)
    file = open('files/temp_'+category+'_data.json','w')
    for utt in tqdm(corpus['train']):
        text = utt['review_body']

        product = utt['product_id']
        if product in product_values:
            product_id = product_values[product]
        else:
            product_id = product_idx
            product_values[product] = product_idx
            product_idx += 1

        customer = utt['customer_id']
        if customer in customer_values:
            customer_id = customer_values[customer]
        else:
            customer_id = customer_idx
            customer_values[customer] = customer_idx
            customer_idx += 1

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
    json.dump(product_values, open("product_values.json",'w'))
    json.dump(customer_values, open("customer_values.json",'w'))
    
for category in category_values.keys():
    os.system('../terashuf/terashuf < files/temp_'+category+'_data.json > files/'+category+'_data.json')
    os.system('rm files/temp_'+category+'_data.json')
    
print(i)
json.dump(i, open("data_info.json",'w'))
