#%%
#import libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', None)

# %%
#Load dataset
df_train = pd.read_csv('df_train_processed.csv')

# %%
df_train.drop(columns=['Unnamed: 0'], inplace = True)

# %%
df_train.info()

#%%
cars = df_train.drop("price", axis=1)
cars_price = df_train["price"].copy()

#%%
cars_num = cars.select_dtypes(include=['float64']).columns
cars_cat = cars.select_dtypes(include=['object']).columns

#%%
full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), cars_num),
        ("cat", OneHotEncoder(), cars_cat)
    ])

#%%
model = Pipeline(steps=[
    ('pre_proc', full_pipeline),
    ('forest_regressor', RandomForestRegressor(random_state=42))
])


# %%
# param_grid = [
#     {'forest_regressor__n_estimators': [3, 5, 10, 20, 30], 
#       'forest_regressor__max_features': [2, 4, 6, 8, 10]}
#      ]

# # %%
# grid_search = GridSearchCV(model, param_grid, cv = 5,
#                            scoring ='neg_mean_squared_error',
#                            return_train_score=True)

# grid_search.fit(cars, cars_price)

# # %%
# grid_search.best_estimator_

# %%
model.fit(cars, cars_price)
print('Trained model')

# %%
# forest_scores = cross_val_score(model, cars, cars_price,
#                                 scoring="neg_mean_squared_error", 
#                                 cv=10)

# forest_rmse_scores = np.sqrt(-forest_scores)

# forest_rmse_scores

# %%
cars_predictions = model.predict(cars.iloc[1000:])
forest_mse = mean_squared_error(cars_price.iloc[1000:], cars_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
# %%
