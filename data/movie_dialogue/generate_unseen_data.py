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

corpus = Corpus(filename=download("movie-corpus"))

character_values = json.load(open("character_values.json",'r'))
movie_values = json.load(open("movie_values.json",'r'))
genre_values = {'fantasy':0}
i = {genre:0 for genre in genre_values.keys()}
files = {genre: open('files/temp_unseen_data.json','w') for genre in genre_values.keys()}
for utt in tqdm(corpus.iter_utterances()):
    text = utt.text
    conv = corpus.get_conversation(utt.conversation_id)
    
    genres = [genre for genre in conv.meta['genre'][2:-2].split('\', \'') if genre != '']
    genre_ids = [genre_values[genre] for genre in genres]
    
    rating = conv.meta['rating']
    if float(rating) <= 10/3:
        rating_id = 1
    elif float(rating) <= 20/3:
        rating_id = 2
    else:
        rating_id = 3
        
    gender = utt.speaker.meta['gender']
    if gender in ['f','F']:
        gender_id = 1
    elif gender in ['m','M']:
        gender_id = 2
    else:
        gender_id = 3
        
    movie = utt.speaker.meta['movie_name'].replace('\"','')
    if movie in movie_values:
        movie_id = movie_values[movie]
    else:
        movie_id = 0
    
    character = movie + '-' + utt.speaker.meta['character_name'].replace('\t\t\t\t\t\t\t  *','').replace('\"','\\"')
    if character in character_values:
        character_id = character_values[character]
    else:
        character_id = 0

    sentences = nltk.tokenize.sent_tokenize(text)
    for sentence in sentences:
        sentence = tokenizer(sentence)['input_ids']
        if len(sentence)<=1024:
            line = "{\"sentence\":" + str(sentence) + "," + \
                   "\"genre\":" + str(genres).replace('\'','\"') + "," + \
                   "\"genre_id\":" + str(genre_ids) + "," + \
                   "\"rating\":" + rating + "," + \
                   "\"rating_id\":" + str(rating_id) + "," + \
                   "\"movie\":\"" + movie + "\"," + \
                   "\"movie_id\":" + str(movie_id) + "," + \
                   "\"gender\":\"" + gender + "\"," + \
                   "\"gender_id\":" + str(gender_id) + "," + \
                   "\"character\":\"" + character + "\"," + \
                   "\"character_id\":" + str(character_id) + "}"
            
            for genre in genres:
                if i[genre] != 0:
                    files[genre].write('\n'+line)
                else:
                    files[genre].write(line)
                i[genre] += 1
for genre in files.keys():
    files[genre].close()
    
for genre in genre_values.keys():
    os.system('../terashuf/terashuf < files/temp_unseen_data.json > files/unseen_data.json')
    os.system('rm files/temp_unseen_data.json')
    
print(i)