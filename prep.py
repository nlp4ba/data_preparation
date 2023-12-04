##### IF YOU WANT TO STORE YOUR DATA, PLEASE DO NOT OVERWRITE THE recipes_raw.csv, create a new one with an other name ######
import pandas as pd

df = pd.read_csv('recipes_raw.csv', delimiter=";")

# Mention that there are the following columns: link, duration, full_duration, ingredients and description.
# Maybe not all columns are displayed with the following print statement 
print(df)