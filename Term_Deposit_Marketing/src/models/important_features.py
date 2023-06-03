# Trying to detect the most informative feature
runfile('src/features/build_features.py')

# Trying a Decision Tree model with a depth of 1 
simple_tree_model = DecisionTreeClassifier(max_depth = 1,
                                    max_leaf_nodes = 2,
                                    max_features = None,
                                    random_state = 42)

simple_tree_cv_results = cross_validate(simple_tree_model, data_train, target_train, cv=5, 
                                        return_train_score = True)

t_scores = simple_tree_cv_results["train_score"]
print("The mean cross-validation training accuracy is: "
      f"{t_scores.mean():.3f} ± {t_scores.std():.3f}")
scores = simple_tree_cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} ± {scores.std():.3f}")

simple_tree_model.fit(data_train, target_train)
plot_tree(simple_tree_model)

print(f'\nThe feature that yields the most information gain is {columns[3]}')

this_transformed_column = transformed_df[columns[3]]
this_column = np.array(dataframe['duration'])
scaler = StandardScaler()
this_retransformed_column = scaler.fit_transform(this_column.reshape(-1,1))
print(f'The average call duration is {int(scaler.mean_)} sec\n',
      f'The standard deviation is {int(scaler.var_**0.5)} sec')

threshold = scaler.inverse_transform(np.array([0.469]*40000).reshape(-1,1))
print(f'The call duration threshold is {threshold[0,0]:.0f} sec')
print(f'The mean cross-validation accuracy was {scores.mean():.3f} just from using that criteria')

# Checking if the scaler StandardScaler fitted the column with the same parameters as ColumnTranformer did
equal = np.array([isclose(this_retransformed_column[i], this_transformed_column[i], abs_tol = 1e-6) for i in range(40000)])
assert equal.all()