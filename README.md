# Marketing Segmentation
Untargeted marketing through call center data would result in an inefficient marketing strategy and low success rate. 
Here, a robust machine learning system is developped that leverages call center data to reduce the cost of marketing efforts.

## Objective
Improving the success rate for calls made to customers for a product offered by the company.

## Dataset
The data comes from direct marketing efforts of a banking institution. 
The dataset contains information about customer demographics, past interactions, and product subscriptions.
The target label indicates whether the prospective customer subscribed to the offer.

## Methodology
A combination of machine learning and statistical techniques is used to predict the probability of success of a marketing effort based on the prospective client attributes and past interactions. This includes:
- Under-sampling of a severely imbalanced dataset and usage of cross-validation techniques.
- Data Scaling and One-Hot Encoding
- Utilizing Decision Trees, Random Forests and XGBoost.
- Hyperparameter tuning of XGBoost with Optuna
- Determining segments of customers that should be prioritized

## Results
The most efficient model, XGBoost, resulted in 0.89 F1-score on the test set after application of cross-validation techniques for hyperparameter selection.
Other models such as Logistic Regression and Random Forests also yield good results.

## Conclusion
Through this marketing segmentation approach, distinct customer segments with unique characteristics were identified. 
This enables targeted marketing strategies and personalized communication.

