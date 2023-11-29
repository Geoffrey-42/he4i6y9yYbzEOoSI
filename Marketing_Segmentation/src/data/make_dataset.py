runfile('setup.py')

path = 'data/raw/'
dataframe = pd.read_csv(path+"term-deposit-marketing-2020.csv")

target_name = 'y'
target = dataframe[target_name]
data = dataframe.drop(columns = [target_name])

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

print('There are %s costumer datapoints'%len(data))
print('\nThere are %s features, which are:'%len(data.columns))
print(set(data.columns))

print('\nThe numerical features are:')
print(numerical_columns)

print('\nThe categorical features are:')
print(categorical_columns)

print('\nThe 5 first customers data are:')
print(data.head())