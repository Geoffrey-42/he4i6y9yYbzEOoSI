runfile('src/data/make_dataset.py')

# Discriminating the heterogeneous features
print('\nDescription of the numerical features in the data:')
print(data.describe())
non_binary_categorical_columns = {}
binary_categorical_columns = []
for category in categorical_columns:
    nb_cat = pd.Series.nunique(data[category])
    if nb_cat > 2:
        non_binary_categorical_columns[category] = f' ({nb_cat} values found)'
    else:
        binary_categorical_columns.append(category)

print('\nThe binary categorical columns are:')
for cat in binary_categorical_columns:
    print(cat, ' (2 values found)')
print('\nThe other categorical columns are:')
for key, value in non_binary_categorical_columns.items():
    print(key, value)
non_binary_categorical_columns = list(non_binary_categorical_columns.keys())

# Encoding the categorical features and scaling the numerical features
ColumnTransformer_ = make_column_transformer(
    (StandardScaler(), numerical_columns),
    (OneHotEncoder(), categorical_columns),
    remainder='passthrough',
    n_jobs = 4)

Transformed_Columns = ColumnTransformer_.fit_transform(data)
columns = ColumnTransformer_.get_feature_names_out()
print(f'The transformed features names are\n{columns}\n')
transformed_df = pd.DataFrame(Transformed_Columns.toarray(), columns = columns)

# Dropping the "no" columns resulting from the one-hot encoding of binary categorical features
columns_to_drop = []
for feature in binary_categorical_columns:
    columns_to_drop.append('onehotencoder__'+feature+'_no')
transformed_df.drop(columns = columns_to_drop, inplace = True)

print(f'The transformed dataset has a shape of {transformed_df.shape}')
print(f'A total of {len(columns)-transformed_df.shape[1]} useless encoded features were removed')

transformed_target = LabelEncoder().fit_transform(target)

# Undersampling the data to obtain balance regarding to the target
print('The data is unbalanced')
print(f'The proportion of the minority class is {sum(transformed_target)/transformed_target.shape[0]:.2f}\n')

# undersample = CondensedNearestNeighbour(n_neighbors=1)
undersample = RandomUnderSampler()
X, y = undersample.fit_resample(transformed_df, transformed_target)

print('The dataset resulting from undersampling has a shape of', X.shape, 
      '\nThe label array resulting from undersampling has a shape of', y.shape)
print(f'The proportion of the minority class is now {sum(y)/y.shape[0]:.2f}\n')

data_train, data_test, target_train, target_test = train_test_split(
    np.array(X), y, test_size = 0.2, stratify = y, random_state=42)

print('The data resulting from undersampling is now divided into a train and a test set 80/20%')
print('The train set and label array have a shape of ', data_train.shape, target_train.shape)
print('The test set and label array have a shape of ', data_test.shape, target_test.shape)