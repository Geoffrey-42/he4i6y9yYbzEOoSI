# Training three different models (Logistic Regression, Decision Tree, XGBoost)
# 5 fold cross-validation 
runfile('src/features/build_features.py')

# Trying a Logistic Regression model (works)
logis_model = LogisticRegression(penalty = 'l2',
                                 C = 1.0,
                                 solver = 'newton-cholesky', # sag and saga solvers can be tried as well
                                 max_iter = 100,
                                 n_jobs = 4)

logis_cv_results = cross_validate(logis_model, data_train, target_train, cv=5, 
                                  return_train_score = True)

t_scores = logis_cv_results["train_score"]
print("The mean cross-validation training accuracy is: "
      f"{t_scores.mean():.3f} ± {t_scores.std():.3f}")
scores = logis_cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} ± {scores.std():.3f}")

# Trying a shallow Decision Tree model (works)
tree_model = DecisionTreeClassifier(max_depth = 3,
                                    max_leaf_nodes = 20,
                                    max_features = None,
                                    random_state = 42)

tree_cv_results = cross_validate(tree_model, data_train, target_train, cv=5, 
                                  return_train_score = True)

t_scores = tree_cv_results["train_score"]
print("The mean cross-validation training accuracy is: "
      f"{t_scores.mean():.3f} ± {t_scores.std():.3f}")
scores = tree_cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} ± {scores.std():.3f}")
tree_model.fit(data_train, target_train)
plot_tree(tree_model)
print(f'The selected features in this tree were:\n{list(transformed_df.iloc[:,[3,30,37,25,29]].columns)}')
print('From the decision tree, for clients with a call duration below the threshold\n' 
       'if the last call was in April and if the client has no housing loan, or if last call was in March\n'
        'y is more likely to be 1')

# Trying a XGBoost model
GridSearch_ = 0 # Change to 1 to run the GridSearch (takes a few minutes to run)
if GridSearch_: 
    model = XGBClassifier(random_state = 42)
    parameters = {'n_estimators':list(range(1,501,100)), 
                  'max_depth':list(range(1,11,1)), 
                  'learning_rate':[round(10**(i/10),3) for i in range(-20, 0, 5)]}
    clf = GridSearchCV(estimator = model, param_grid = parameters, cv = 5, return_train_score = True)
    clf.fit(data_train, target_train)
    print('The best parameters in the specified grid are found to be\n', clf.best_params_)
    print(f"Mean training accuracy: {clf.cv_results_['mean_train_score'][clf.best_index_]:0.3f}")
    print(f'Cross-validation accuracy: {clf.best_score_:0.3f}')

# Best parameters based on the GridSearch results:
best_params = {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}

# Trying the XGBoost model with these parameters:
n_estimators = best_params['n_estimators']
max_depth = best_params['max_depth']
learning_rate = best_params['learning_rate']
model = XGBClassifier(n_estimators = n_estimators, 
                             max_depth = max_depth, 
                             learning_rate = learning_rate, 
                             random_state = 42)

cv_results = cross_validate(model, data_train, target_train, cv=5,
                            return_train_score = True)
t_scores = cv_results["train_score"]
print("The mean training accuracy is: "
      f"{t_scores.mean():.3f} ± {t_scores.std():.3f}")
scores = cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} ± {scores.std():.3f}")

# Overfitting test
overfitting_test = 1 # Change to 1 to see results
if overfitting_test:
    print('An overfitting test with variable max_depth is now running')
    training_accuracies = []
    validation_accuracies = []
    for max_depth in range(1,11):
        model = XGBClassifier(n_estimators = n_estimators, 
                                     max_depth = max_depth, 
                                     learning_rate = learning_rate, 
                                     random_state = 42)
    
        cv_results = cross_validate(model, data_train, target_train, cv=5,
                                    return_train_score = True)
    
        t_scores = cv_results["train_score"]
        training_accuracies.append(t_scores.mean())
        scores = cv_results["test_score"]
        validation_accuracies.append(scores.mean())
    fig, ax = plt.subplots(1,1)
    ax.plot(list(range(1,11)), training_accuracies, color = 'blue', label = 'Training Accuracy')
    ax.plot(list(range(1,11)), validation_accuracies, color = 'red', label = 'Validation Accuracy')
    ax.legend(fontsize = 12)
    ax.set_xlabel('Max_depth')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Overfitting test')
    plt.show()

print('The cross-validation results for each fold are provided below with more details')

best_params = {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}
n_estimators = best_params['n_estimators']
max_depth = best_params['max_depth']
learning_rate = best_params['learning_rate']

model = XGBClassifier(n_estimators = n_estimators, 
                             max_depth = max_depth, 
                             learning_rate = learning_rate, 
                             random_state = 42)

def Kfold_crossvalidation(partition):
    accuracy = 0
    n = 0
    for train, test in partition:
        model.fit(data_train[train], target_train[train])
        target_train_predict = model.predict(data_train[test])
        accuracy += model.score(data_train[test], target_train[test])
        n += 1
        print(classification_report(target_train[test], target_train_predict))
    accuracy *= 1/n
    print(f'cross validation accuracy = {accuracy}')

skf = StratifiedKFold(5, shuffle = False)
Kfold_crossvalidation(skf.split(data_train, target_train))