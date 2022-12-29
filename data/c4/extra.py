import os

sites =  ['frontiersin.org','chicagotribune.com','link.springer.com',
          'aljazeera.com','instructables.com','npr.org',
          'dailymail.co.uk','csmonitor.com','baltimoresun.com','city-data.com','unseen']

for site in sites:
    os.system('../terashuf/terashuf < files/temp_'+site+'_data.json > files/'+site+'_data.json')
    os.system('rm files/temp_'+site+'_data.json')