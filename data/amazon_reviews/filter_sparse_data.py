import json

with open('../features.txt') as f:
    FEATURES = json.loads(f.read())
    
categories = ["video_games", "pet_products", "grocery", "home", "electronics",  "beauty", "baby", "automotive", "apparel", "books"]

for category in categories:
    
    feature_dict = {}
    for i in range(FEATURES['amazon_reviews']['product']):
        feature_dict[i] = 0
    
    with open('files/'+category+'_data.json','r') as f: 
        for line in f:
            data = json.loads(line)
            feature_dict[data['product_id']] += 1

    top_features = sorted(feature_dict, key=feature_dict.get, reverse=True)[:10]

    count = 0
    with open('files/'+category+'_data_filtered.json','w') as f_filtered: 
        with open('files/'+category+'_data.json','r') as f: 
            for line in f:
                data = json.loads(line)
                if data['product_id'] in top_features:
                    if count != 0:
                        f_filtered.write('\n'+line)
                    else:
                        f_filtered.write(line)
                    count += 1
                    
    print(category)
    print(top_features)
    print(count)
    print()