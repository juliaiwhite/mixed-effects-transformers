import json

with open('../features.txt') as f:
    FEATURES = json.loads(f.read())
    
genres = ['unseen', 'action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama', 'family', 'film-noir', 'history', 'horror', 'music', 'musical', 'mystery', 'romance', 'sci-fi', 'short', 'sport', 'thriller', 'war', 'western']

for genre in genres:
    feature_dict = {}
    for i in range(FEATURES['movie_dialogue']['movie']):
        feature_dict[i+1] = 0
    
    with open('files/'+genre+'_data.json','r') as f: 
        for line in f:
            data = json.loads(line)
            feature_dict[data['movie_id']] += 1

    top_features = sorted(feature_dict, key=feature_dict.get, reverse=True)[:10]

    count = 0
    with open('files/'+genre+'_data_filtered.json','w') as f_filtered: 
        with open('files/'+genre+'_data.json','r') as f: 
            for line in f:
                data = json.loads(line)
                if data['movie_id'] in top_features:
                    if count != 0:
                        f_filtered.write('\n'+line)
                    else:
                        f_filtered.write(line)
                    count += 1
                    
    if genre in ['action', 'adventure', 'comedy', 'crime', 'drama', 'horror', 'mystery', 
                 'romance', 'sci-fi', 'thriller','unseen']:
        print(genre)
        print(top_features)
        print(count)
        print()