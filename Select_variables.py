#%%
# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
pd.set_option('display.max_columns', None)

# %%
#Load dataset
df_cars = pd.read_csv('vehicles.csv')
#%%
df_cars.tail(5)

#%%
#See the name of columns and select only we're used
df_cars.columns
# %%
df_cars = df_cars[['price', 'year', 'manufacturer', 'model',
'odometer', 'transmission', 'type']]
# %%
#drop na informations
df_cars = df_cars.dropna()
df_cars = df_cars.query('odometer > 1000 and odometer <= 300000')
df_cars = df_cars.query('price > 0 and price <= 350000')
df_cars = df_cars.query('year > 1980')
df_cars.drop_duplicates(inplace=True)
df_cars.info()
# %%
#These are the columns I'm going to use to create a model.
#Now, I'm going to create a train and test dataset 
# %%
df_cars.reset_index(inplace = True)
# %%
df_cars["price_cat"] = pd.cut(df_cars['price'],
                        bins = [0, 10000, 20000, 30000,
                        40000, 50000, 60000, np.inf],
                        labels = [1,2,3,4,5,6,7])
# %%
sss = StratifiedShuffleSplit(n_splits = 5, 
                            test_size = 0.2, 
                            random_state = 42)
# %%
for train_index, test_index in sss.split(df_cars, df_cars['price_cat']):
    print('Train:', train_index, 'Test:', test_index)
    strat_train_set = df_cars.loc[train_index]
    strat_test_set = df_cars.loc[test_index]
# %%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("price_cat", axis=1, inplace=True)

# %%
# saves files in .csv
strat_train_set.to_csv('strat_train_set.csv')

# %%
strat_test_set.to_csv('strat_test_set.csv')

print('Dataset saved successfully!')