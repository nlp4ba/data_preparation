##### IF YOU WANT TO STORE YOUR DATA, PLEASE DO NOT OVERWRITE THE recipes_raw.csv, create a new one with an other name ######
import pandas as pd
import re

df = pd.read_csv('recipes_raw.csv', delimiter=";"   )

# Mention that there are the following columns: link, duration, full_duration, ingredients and description.
# Maybe not all columns are displayed with the following print statement 

#Delete Link
df = df.drop(columns="link")

#Format Duration in min
df = df.replace({'duration': ' Min.'}, {'duration': ''}, regex=True)
df['duration'] = df['duration'].astype(int)

#Format Full Duration in min
def replaceTime(x):
    result = re.findall(r'(\d+) Stunde|(\d+) Minuten', x)
    print("INPUT", x, result)
    if(len(result) == 1):
        return result[0][1]
    elif(len(result) == 2):
        return int(result[0][0]) * 60 + int(result[1][1])
    else:
        return None 
df['full_duration'] =  [replaceTime(x) for x in df['full_duration']]

#Rename difficulty categories
df = df.replace({'difficulty': 'simpel'}, {'difficulty': 'easy'}, regex=True)
df = df.replace({'difficulty': 'normal'}, {'difficulty': 'middle'}, regex=True)
df = df.replace({'difficulty': 'pfiffig'}, {'difficulty': 'hard'}, regex=True)

print(df)