
#This is Project 2 for Text Analytics - Spring 2020
#Written by Nicholas Cejda

import pandas as pd
import json
#import spacy
import os

#Step 1 is to read in the metadata.csv file and take a look around.
#Let's read it in with the read.csv function from pandas.

metadata_csv_path = os.path.join(os.path.abspath(os.path.curdir), "docs\\CORD-19-research-challenge\\metadata.csv")

metaCSV = pd.read_csv(metadata_csv_path)
metaCSV.info()
#For sure we will need the cord_uid, sha, and pmcid, to identify them later, as well as the title, abstract, authors, and journal

#Lets do some cleanup on this dataset. I can actually subset this list to a random 10%, then search through the
#JSON files to find matching sha or pmcid, and use that as my test subset.
#Or, I can read in all the JSON files, and subset the 10% from there.
#Dr. Grant mentioned we only need to search through the comm_use_subset -> pdf_json files. We can essentially ignore all else.
#In that case, all we need is files with the sha as the ID.


metaCSV = metaCSV[['cord_uid', 'sha', 'pmcid', 'title', 'abstract', 'authors', 'journal']]

#Need to remove items with no title or abstract or sha value
metaCSV.dropna(inplace=True, subset = {'title', 'abstract', 'sha'})
metaCSV.info() #33536 entries.


random_sample = metaCSV.sample(n=3)
random_sample.info()
random_sample['sha']

filename = str(random_sample['sha'][0]) + ".json"




#We want to drop items with no sha AND no pmcid, as they will be impossible to find in the JSON files.

metaCSV.dropna(inplace=True, thresh=2, subset = {'sha', 'pmcid'})
metaCSV.info() #28802 entries

metaCSV.dropna(inplace=True, subset = {'sha', 'pmcid'})
metaCSV.info() #28802 entries

metaCSV = metaCSV[metaCSV['sha'].notna() & metaCSV['pmcid'].notna()]
metaCSV.info()
for i in range (0,5):
    print(metaCSV.iloc[[i]])


metaCSV.drop(metaCSV[metaCSV['sha'].notna() & metaCSV['pmcid'].notna()].index, inplace=True) #13401 entries
