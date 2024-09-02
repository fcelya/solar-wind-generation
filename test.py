from test_module import test_results
import pandas as pd


TEST_HORIZON = 24*365
RESULTS = 'results/sarimax_solar_2024-05-03_07-45-07.csv'
VARIABLE = 'eolica'

def read_dataset(name='data/factor_capacidad_conjunto.csv',data='solar'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    df = df.rename(columns={
        data: 'y'
    })
    return df[['y']]

df_pred = pd.read_csv(RESULTS, index_col=0, parse_dates=True)
df_true = read_dataset(data=VARIABLE)

# print('\nTEST:')
test_results(df_true['y'], df_pred['y'], test_horizon=TEST_HORIZON)


