# Running the previous models on test data
runfile('src/models/train_model.py')

logis_model.fit(data_train, target_train)
target_test_predict = logis_model.predict(data_test)
print('\nFor the Logistic Regression model, the test results are:\n')
print(classification_report(target_test, target_test_predict))

tree_model.fit(data_train, target_train)
target_test_predict = tree_model.predict(data_test)
print('For the Decision Tree model, the test results are:\n')
print(classification_report(target_test, target_test_predict))

model.fit(data_train, target_train)
target_test_predict = model.predict(data_test)
print('For the XGBoost model, the test results are:\n')
print(classification_report(target_test, target_test_predict))


