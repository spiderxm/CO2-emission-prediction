import numpy as np
import pandas

df = pandas.read_csv('data/data.csv')
data = pandas.DataFrame(
    columns=['Country', 'Year', 'CO2 Emission Per Capita', 'electricity', 'forest_area', 'gdp', 'gdp_per_capita',
             'urban_population_pct', 'population_growth_pct', 'population', 'renewable_energy_consumption_pct',
             'energy_used', 'urban_population_growth_pct'])

for index, row in df.iterrows():
    for i in range(2005, 2020):
        if row[str(i)] is not None and row[str(i)] is not float(0) and type(row[str(i)]) is not str:
            data.loc[len(data.index)] = [row["Country Name"], i, row[str(i)], np.nan, np.nan, np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan, np.nan, np.nan]

feature_names = ['electricity', 'forest_area', 'gdp', 'gdp_per_capita', 'urban_population_pct', 'population_growth_pct',
                 'population', 'renewable_energy_consumption_pct', 'energy_used', 'urban_population_growth_pct']


def add_feature_data(feature_name):
    feature_data = pandas.read_csv('data/{}.csv'.format(feature_name))

    for index, row in data.iterrows():
        row = row.copy()
        data.loc[index, feature_name] = float(
            feature_data.loc[feature_data["Country Name"] == row["Country"]][
                str(row["Year"])])


for feature in feature_names:
    add_feature_data(feature_name=feature)

data.to_csv("data/data_file.csv")
