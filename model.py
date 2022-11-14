import pickle

import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import linear_model

feature_names = ['electricity', 'forest_area', 'gdp', 'gdp_per_capita', 'urban_population_pct', 'population_growth_pct',
                 'population', 'renewable_energy_consumption_pct', 'energy_used', 'urban_population_growth_pct']

df = pandas.read_csv('data/data_file.csv')

# Filling Empty Values
for feature in feature_names:
    mode = df[feature].mode()
    df[feature].fillna(mode, inplace=True)

df = df.dropna()

y = df['CO2 Emission Per Capita']
x = df.drop(columns=['CO2 Emission Per Capita'])

x = x.drop(columns=['Country'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

regr = RandomForestRegressor(n_estimators=300, max_features='sqrt', max_depth=5, random_state=18)
regr.fit(x_train, y_train)

# random forest regressor
pred = regr.predict(x_train)
res = r2_score(y_train, pred)
print(res)

pred = regr.predict(x_test)
res = r2_score(y_test, pred)
print(res)

pickle.dump(regr, open('models/random_forest_regressor_model.pkl', 'wb'))

# multiple linear regression

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

pred = regr.predict(x_train)
res = r2_score(y_train, pred)
print(res)
pred = regr.predict(x_test)
res = r2_score(y_test, pred)
print(res)

pickle.dump(regr, open('models/linear_model.pkl', 'wb'))
