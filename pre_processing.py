#%%
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns

pd.set_option('display.max_columns', None)

# %%
#Load dataset
df_train = pd.read_csv('strat_train_set.csv')
# %%
df_train.drop(columns=['Unnamed: 0'], inplace = True)
# %%
df_train.info()
# %%
sns.set(rc={"figure.figsize":(6, 10)}) 
sns.countplot(y = df_train['manufacturer'],
            order = df_train['manufacturer'].value_counts().index)
# %%
df_train.manufacturer.value_counts()

# %%
other = ['lexus', 'audi', 'kia', 'acura', 'cadillac', 'chrysler',
'mazda', 'infiniti', 'buick', 'lincoln', 'volvo', 'mitsubishi',
'mini', 'jaguar', 'rover', 'pontiac', 'porshe', 'alfa-romeo',
'tesla', 'mercury', 'saturn', 'fiat', 'harley-davidson',
'ferrari', 'aston-martin', 'datsun', 'land rover']
#%%
df_train['manufacturer'] = df_train['manufacturer'].replace(other,'others')
df_train['manufacturer'] = df_train['manufacturer'].replace(['ram','dodge'],'dodge-ram')
# %%
df_train['year'] = 2021 - df_train['year']
# %%
df_train.info()
# %%
df_train.drop(columns=['index'], inplace = True)

# %%
# saves files in .csv
df_train.to_csv('df_train_processed.csv')

# %%
print('Saved files!')
# %%
