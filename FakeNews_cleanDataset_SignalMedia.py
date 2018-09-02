import os
import jsonlines
import pandas as pd
import numpy as nd

os.chdir('C:\\Users\\Jeh\\Desktop\\Course Stuff\\CSE6242 - Data and Visual Analytics\\Project\\FakeNews')

s=0
a=[]
with jsonlines.open('realnews.jsonl') as reader:
    for obj in reader:
        a.append(obj)

#data=pd.DataFrame(columns=['id'], index = range(len(a)), data='0')
#data['text']='0'
#data['source']='0'
#data['media_type']='0'
#data['published_timestamp']='0'
#data['title']='0'
           
#for index,line in enumerate(a):
#    data['text'][index]=str(line['content'])
#    data['source'][index]=str(line['source'])
#    data['media_type'][index]=str(line['media-type'])
#    data['published_timestamp'][index]=str(line['published'])
#    #data['source'][index]=str(line['source'])
#    data['title'][index]=str(line['title'])
    
data=pd.DataFrame(a)
unique_sources=list(data['source'].unique())
#unique_sources.to_csv(unique_sources,'unique_sources.csv')
unique_sources_df=pd.DataFrame(unique_sources)
unique_sources_df.to_csv('unique_sources.csv')



data_real=data[data['source']=='Reuters'] #3898
data_real=pd.concat([data_real,data[data['source']=='New York Times']]) #322
data_real=pd.concat([data_real,data[data['source']=='Economist']]) #1
data_real=pd.concat([data_real,data[data['source']=='BBC']]) #830
data_real=pd.concat([data_real,data[data['source']=='BBC News - Asia']]) #48
data_real=pd.concat([data_real,data[data['source']=='BBC News - Business']]) #33
data_real=pd.concat([data_real,data[data['source']=='Guardian.co.uk']]) #1077
data_real=pd.concat([data_real,data[data['source']=='Washington Post']]) #820
data_real=pd.concat([data_real,data[data['source']=='Time']]) #227
data_real=pd.concat([data_real,data[data['source']=='TIME']]) #118
data_real=pd.concat([data_real,data[data['source']=='Wall Street Journal UK']]) #1
data_real=pd.concat([data_real,data[data['source']=='PBS']]) #37
data_real=pd.concat([data_real,data[data['source']=='latimes.com - Los Angeles Times']]) #320
data_real=pd.concat([data_real,data[data['source']=='CNN']]) #595
data_real=pd.concat([data_real,data[data['source']=='CNBC']]) #948
data_real=pd.concat([data_real,data[data['source']=='Seattle Times']]) #287
data_real=pd.concat([data_real,data[data['source']=='Denver Post']]) #288
data_real=pd.concat([data_real,data[data['source']=='POLITICO']]) #1
data_real=pd.concat([data_real,data[data['source']=='Hindustan Times']]) #1688
data_real=pd.concat([data_real,data[data['source']=='Bloomberg']]) #2833

data_real.to_csv('real_dataset.csv')   