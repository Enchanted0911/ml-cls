
import pandas as pd
# import numpy as np
# import seaborn as sns
# sns.set()
import datetime
from dateutil.relativedelta import *
from pandas import DataFrame, Series
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# %matplotlib notebook

# table = pd.read_csv('raw_sample.csv')
table = pd.read_csv('raw2.csv')
table = table.set_index('vin')
table['dateIndex'] = table.apply(lambda _: '', axis=1)
table['vinchar1'] = table.apply(lambda _: '', axis=1)
table['vinchar2'] = table.apply(lambda _: '', axis=1)
table['vinchar3'] = table.apply(lambda _: '', axis=1)
table['vinchar4'] = table.apply(lambda _: '', axis=1)
table['vinchar5'] = table.apply(lambda _: '', axis=1)
table['vinchar6'] = table.apply(lambda _: '', axis=1)
table['vinchar7'] = table.apply(lambda _: '', axis=1)
table['vinchar8'] = table.apply(lambda _: '', axis=1)
table['vinchar9'] = table.apply(lambda _: '', axis=1)
table['vinchar10'] = table.apply(lambda _: '', axis=1)
table['vinchar11'] = table.apply(lambda _: '', axis=1)
table['vinchar12'] = table.apply(lambda _: '', axis=1)
table['vinchar13'] = table.apply(lambda _: '', axis=1)
table['vinchar14'] = table.apply(lambda _: '', axis=1)
table['vinchar15'] = table.apply(lambda _: '', axis=1)
table['vinchar16'] = table.apply(lambda _: '', axis=1)
table['vinchar17'] = table.apply(lambda _: '', axis=1)

df = pd.DataFrame()
# table = table[['dateIndex',  'vinchar1', 'vinchar2',
#                        'vinchar3', 'vinchar4', 'vinchar5', 'vinchar6', 'vinchar7', 'vinchar8',
#                        'vinchar9', 'vinchar10', 'vinchar11', 'vinchar12', 'vinchar13', 'vinchar14',
#                        'vinchar15', 'vinchar16', 'vinchar17']]
# df.drop(df.index, inplace=True)

startdt = datetime.datetime.strptime("1980-01-01", "%Y-%m-%d")
for index, row in table.iterrows():
    # print(index, row)
    for i, c in enumerate(index, start=1):
        # print("vin char", i, " is ", c)
        # Convert VIN character to number for ML use
        try:
            vinchar_int = int(c)
        except ValueError:
            vinchar_int = ord(c) - ord('A') + 10
        row[1 + i] = vinchar_int
    productDate = str(row["productDate"])[0:4] + "-" + str(row["productDate"])[4:6]
    try:
        datedt = datetime.datetime.strptime(productDate, "%Y-%m")
    except Exception:
        continue
    dateIndex = relativedelta(datedt,startdt).years * 12 + relativedelta(datedt,startdt).months
    row["dateIndex"] = dateIndex
    # row["vin"] = index
    df = df.append(row)
    # print(index)

# print(df)
df.to_csv("processed_data.csv")