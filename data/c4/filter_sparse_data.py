import json

with open('../features.txt') as f:
    FEATURES = json.loads(f.read())
    
sites =  ['frontiersin.org','chicagotribune.com','link.springer.com',
          'aljazeera.com','instructables.com','npr.org',
          'dailymail.co.uk','csmonitor.com','baltimoresun.com','city-data.com']

for site in sites:
    
    feature_dict = {}
    for i in range(FEATURES['c4']['url']):
        feature_dict[i] = 0
    
    with open('files/'+site+'_data.json','r') as f: 
        for line in f:
            data = json.loads(line)
            feature_dict[data['url_id']] += 1

    top_features = sorted(feature_dict, key=feature_dict.get, reverse=True)[:10]

    count = 0
    with open('files/'+site+'_data_filtered.json','w') as f_filtered: 
        with open('files/'+site+'_data.json','r') as f: 
            for line in f:
                data = json.loads(line)
                if data['url_id'] in top_features:
                    if count != 0:
                        f_filtered.write('\n'+line)
                    else:
                        f_filtered.write(line)
                    count += 1
                    
    print(site)
    print(top_features)
    print(count)
    print()