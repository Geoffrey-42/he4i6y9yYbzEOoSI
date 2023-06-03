# Trying to identify a costumer segment more likely to be successful
runfile('src/features/build_features.py')

# Logistic Regression model
segment_df = transformed_df.iloc[:,5:23]
print(f'The features considered are now\n{list(segment_df.columns)}')
segment_data_train = data_train[:,5:23]

logis_model = LogisticRegression(penalty = 'l2',
                                 C = 1.0,
                                 solver = 'newton-cholesky', # sag and saga solvers can be tried as well
                                 max_iter = 100,
                                 n_jobs = 4)

logis_cv_results = cross_validate(logis_model, segment_data_train, target_train, cv=5, 
                                  return_train_score = True)

t_scores = logis_cv_results["train_score"]
print("The mean cross-validation training accuracy is: "
      f"{t_scores.mean():.3f} ± {t_scores.std():.3f}")
scores = logis_cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} ± {scores.std():.3f}")

# Decision Tree model
segment_tree_model = DecisionTreeClassifier(max_depth = None,
                                            max_features = None,
                                            random_state = 42)

segment_tree_cv_results = cross_validate(segment_tree_model, segment_data_train, target_train, cv=5, 
                                         return_train_score = True)

t_scores = segment_tree_cv_results["train_score"]
print("The mean cross-validation training accuracy is: "
      f"{t_scores.mean():.3f} ± {t_scores.std():.3f}")
scores = segment_tree_cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} ± {scores.std():.3f}")
print('\nThe segment of costumers does not provide enough information to yield a conclusion on the target y.')
print('No costumer segment should be prioritized')