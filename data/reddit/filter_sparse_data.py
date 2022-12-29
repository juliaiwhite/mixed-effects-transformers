import json

with open('../features.txt') as f:
    FEATURES = json.loads(f.read())
    
subreddits =  ['aww', 'todayilearned', 'apple', 'pokemontrades', 'relationship_advice', 'DebateReligion', 'worldnews', 'nba', 'Naruto', 'hiphopheads']

for subreddit in subreddits:
    
    feature_dict = {}
    for i in range(FEATURES['reddit']['speaker']):
        feature_dict[i] = 0
    
    with open('files/'+subreddit+'_data.json','r') as f: 
        for line in f:
            data = json.loads(line)
            feature_dict[data['speaker_id']] += 1

    top_features = sorted(feature_dict, key=feature_dict.get, reverse=True)[:10]

    count = 0
    with open('files/'+subreddit+'_data_filtered.json','w') as f_filtered: 
        with open('files/'+subreddit+'_data.json','r') as f: 
            for line in f:
                data = json.loads(line)
                if data['speaker_id'] in top_features:
                    if count != 0:
                        f_filtered.write('\n'+line)
                    else:
                        f_filtered.write(line)
                    count += 1
                    
    print(subreddit)
    print(top_features)
    print(count)
    print()