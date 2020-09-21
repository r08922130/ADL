
import pandas as pd
import unicodedata
import re


path = 'release/dev/ca_data/300351813.pdf.xlsx'

data = pd.read_excel(path, skiprows = 0, sheet_name = 0) # "data" are all sheets as a dictionary

print("###########Columns#############")
print(data.columns)
dic = {}
for i, column in enumerate(data.columns):
    dic[column] = i 
print(dic) 
print("###########Head#############")
print(data.head())
#print(data.tail())
print(data.iloc()[ 1 ,:] )

#(row , column name)
print(unicodedata.normalize("NFKC",re.sub('＊|\*|\s+', '', data.iloc()[ 0, dic['Text']] )))

