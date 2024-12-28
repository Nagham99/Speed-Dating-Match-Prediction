# Speed-Dating-Match-Prediction
This project predicts the outcome of speed dating sessions to build a recommendation system for better matches. A binary classification task, it handles missing data, optimizes a pipeline for preprocessing and model training, and addresses class imbalance, ensuring accurate probability predictions for successful matches.
### Problem Formulation:

#### Input:
 Information about a speed dating session, including profiles of two people.

#### Output:
Probability (0-1, float) that the dating session will lead to a successful match

#### Data Mining Function Required:
This is a binary classification and prediction problem where the goal is to classify instances into two categories: match or no match.The task at hand involves predicting whether two individuals will be a match in speed dating events based on various attributes and interactions.

## Challenges

| Challenge                   | Description                                                                                                                    |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Dealing with missing values | The dataset contains missing values that need to be handled appropriately, either through imputation or deletion, before training the models.                                           |
| Imbalanced Classes          | The dataset exhibits class imbalance, with significantly with significantly more instances of class 0 (no match) compared to class 1 (match). This imbalance may need to be addressed to prevent the model from being biased towards the majority class.the imbalanced classes which could affect model performance.                                     |
| Model Selection and Tuning  | Choosing an appropriate classification algorithm and tuning its hyperparameters to optimize performance can be challenging and time-consuming.                                              |
| Computational Resources     | Training complex models or performing exhaustive hyperparameter search methods can be computationally intensive and require adequate resources.                                             |
| Interpretability vs. Performance | Balancing between model interpretability and performance can be challenging, as some models offer better interpretability at the expense of predictive accuracy.           |
| Overfitting and Generalization | Preventing models from overfitting to the training data and ensuring good generalization performance on unseen data is crucial for model robustness.                               |
| Data Leakage                | Avoiding data leakage, where information from the validation or test set influences model training, is essential for unbiased performance evaluation.                               |
| Scalability                 | Ensuring that the chosen model and its training process are scalable to larger datasets or production environments is important for real-world deployment.                         |
| Model Evaluation Metrics    | Selecting appropriate evaluation metrics that align with the problem objectives and account for class imbalance is essential for accurate model assessment.                     |
| Handling Non-Numeric Data   | Transforming categorical or non-numeric data into a suitable format for modeling requires proper encoding or preprocessing techniques.                                                  |
| Model Interpretability      | Balancing the need for model interpretability with complexity and performance is essential for gaining insights into the model's decision-making process.                       |
| Ensemble Methods            | Exploring ensemble methods to improve model performance by combining the predictions of multiple base models.                                     |
| Time Challenges in Optimization | Grid search, while exhaustive, can become time-consuming, especially with a large hyperparameter space or computationally intensive models.                                         |
| Used Over-sampling          | Applied over-sampling techniques to handle imbalanced data, ensuring that both classes are adequately represented in the training data.                                                |
| Imputer Pipeline            | Employed an imputer pipeline for both numerical and categorical data in the preprocessing step to handle missing values appropriately.                                                  |
| Implemented in Pipeline     | All trials were implemented using pipeline structures to streamline the workflow and ensure reproducibility, scalability, and ease of deployment.                                    |

## Experimental Protocol

1. **Data Loading:** Load the training and test datasets into the environment for analysis and model building.
2. **Data Cleaning:** Perform initial data exploration and identify any missing values or anomalies in the dataset.
3. **Preprocessing:** Apply preprocessing steps to handle missing values, scale numerical features, and encode categorical features using pipelines.
4. **Model Selection:** Experiment with various classification models such as Logistic Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost, and others.
5. **Hyperparameter Tuning:** Utilize different hyperparameter tuning techniques including Grid Search, Random Search, and Bayesian Optimization to optimize model performance.
6. **Model Evaluation:** Evaluate each model's performance using AUROC as the metric on the predicted probability.
7. **Submission Preparation:** Prepare the submission file using the best-performing model to predict the probability of a match in the test dataset and save the results in a CSV file.

8. ### Impact: 
Improving the matching process in speed dating events, leading to higher satisfaction among participants and potentially increasing the success rate of matches.
The implementation of an effective speed dating match prediction system can have several significant impacts:

1. **Enhanced User Experience**: By accurately predicting potential matches, participants in speed dating events can have more meaningful interactions, leading to higher satisfaction and engagement.

2. **Improved Matchmaking**: A reliable recommendation system can facilitate better matches based on compatibility, preferences, and mutual interests, increasing the likelihood of successful connections and subsequent dates.

3. **Time and Resource Efficiency**: With a streamlined matchmaking process, organizers can optimize event logistics, allocate resources more efficiently, and ensure a smoother overall experience for participants.

4. **Data-Driven Insights**: Analyzing the outcomes of speed dating sessions can provide valuable insights into human behavior, preferences, and relationship dynamics, contributing to research in social psychology and interpersonal communication.

5. **Business Opportunities**: A successful speed dating match prediction system can open up opportunities for businesses in the dating industry, including event organizers, dating platforms, and matchmaking services, by offering innovative solutions and attracting a broader audience.

Overall, the development and implementation of a robust speed dating match prediction system can positively impact participants, organizers, researchers, and businesses in the dating industry, leading to improved matchmaking experiences and valuable insights into human relationships.

### Ideal Solution:
A robust binary classification model trained on clean, with carefully selected features and tuned hyperparameters.
In an ideal solution, we would focus on the top-performing models identified in our analysis: XGBoost with Random Search (Trial 8), Random Forest with Random Search (Trial 2), and Random Forest with Grid Search (Trial 3). These models demonstrated competitive performance and robustness in capturing complex relationships within the dataset.

1. **XGBoost with Random Search (Trial 8)**: This model achieved the highest ROC AUC score of 0.9983, indicating its effectiveness in predicting speed dating outcomes. In an ideal solution, we would further optimize this model by expanding the search space for hyperparameters, potentially leading to even better performance.

2. **Random Forest with Random Search (Trial 2)**: With a ROC AUC score of 0.9977, this model also showed promising results. In the ideal solution, we would continue to explore and fine-tune the hyperparameters using random search to uncover additional improvements.

3. **Random Forest with Grid Search (Trial 3)**: Although slightly lower than Trial 2, this model achieved a ROC AUC score of 0.9978, indicating its robustness and effectiveness. In the ideal solution, we would further optimize this model by conducting a comprehensive grid search to fine-tune hyperparameters and potentially enhance performance.

By focusing on these top-performing models and leveraging advanced hyperparameter tuning techniques, such as random search and grid search, we can strive to achieve even higher predictive accuracy for speed dating outcomes. This approach would provide valuable insights for matchmaking algorithms and contribute to enhancing the overall user experience in speed dating events


## Model Tuning and Documentation:
| Trail | Model                         | Reason                                     | Expected Outcome                            | Observations                                   |
|-------|-------------------------------|--------------------------------------------|---------------------------------------------|------------------------------------------------|
| 1     | Logistic Regression           | Baseline model for binary classification  | Obtain initial performance metrics          | Achieved moderate ROC AUC score                |
| 2     | Random Forest                 | Ensemble learning with decision trees      | Improved performance compared to Logistic Regression | Significant increase in ROC AUC score   |
| 3     | Random Forest with Hyperparameter Tuning | Optimization of Random Forest hyperparameters | Further enhancement of performance    | Improved ROC AUC score and identification of optimal hyperparameters |
| 4     | Random Forest with Grid Search| Exhaustive search for optimal hyperparameters | Potential for fine-tuning hyperparameters | Similar performance to Random Forest with Hyperparameter Tuning |
| 5     | Gradient Boosting Classifier  | Sequential model building approach         | Improved performance compared to Random Forest | Enhanced ROC AUC score                        |
| 6     | Gradient Boosting Classifier with Grid Search | Grid search for optimal hyperparameters | Fine-tuning of Gradient Boosting Classifier | Comparable performance to Gradient Boosting Classifier |
| 7     | XGBoost Classifier           | Implementation of XGBoost algorithm       | Expected improvement in performance         | Achieved higher ROC AUC score compared to previous models |
| 8     | XGBoost Classifier with Randomized Search | Randomized search for hyperparameters | Efficient exploration of hyperparameter space | Substantial improvement in ROC AUC score   |
| 9     | XGBoost Classifier with Grid Search | Grid search for hyperparameters      | Fine-tuning of XGBoost Classifier           | Improved ROC AUC score and identification of optimal hyperparameters |
| 10    | LightGBM Classifier           | Utilization of LightGBM algorithm         | Expected enhancement in performance         | Achieved higher ROC AUC score compared to previous models |
| 11    | LightGBM Classifier with Grid Search | Grid search for hyperparameters | Further optimization of LightGBM Classifier | Improved ROC AUC score and identification of optimal hyperparameters |
| 12    | CatBoost Classifier           | Implementation of CatBoost algorithm     | Expected performance improvement             | Achieved ROC AUC score comparable to previous models |
| 13    | CatBoost Classifier with Grid Search | Grid search for hyperparameters | Fine-tuning of CatBoost Classifier          | Improved ROC AUC score and identification of optimal hyperparameters |
| 14    | AdaBoost Classifier           | Ensemble learning with boosting           | Potential for performance improvement       | Lower ROC AUC score compared to other models  |
| 15    | MLP Classifier                | Utilization of neural network model      | Expected performance enhancement             | Achieved significantly higher ROC AUC score compared to AdaBoost Classifier |
| 16    | SVC with Bayesian Optimization| Bayesian optimization for hyperparameters | Efficient exploration of hyperparameter space | Improved ROC AUC score compared to default hyperparameters |
## Model Evaluation and Analysis
| Model                         | ROC AUC Score | Best Parameters                                            | Observation                                                                                        |
|-------------------------------|---------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Random Forest                 | 0.9906        | {'classifier__max_depth': 15, 'classifier__n_estimators': 200}                                     | High ROC AUC score, potential overfitting due to high max depth.                                   |
| Random Forest (Random Search) | 0.9977        | {'classifier__max_depth': 22, 'classifier__n_estimators': 127}                                     | Improved ROC AUC score compared to baseline Random Forest.                                         |
| Random Forest (Grid Search)   | 0.9978        | {'classifier__max_depth': 24, 'classifier__n_estimators': 150}                                     | Slightly improved ROC AUC score compared to Random Search.                                         |
| Gradient Boosting             | 0.9136        | {'classifier__learning_rate': 0.5, 'classifier__max_depth': 10, 'classifier__n_estimators': 200}  | Moderate ROC AUC score, potential for improvement.                                                  |
| Gradient Boosting (Grid Search)| 0.9970       | {'classifier__learning_rate': 0.5, 'classifier__max_depth': 10, 'classifier__n_estimators': 200}  | Comparable ROC AUC score to Random Forest with Grid Search.                                         |
| XGBoost                       | 0.9847        | {'classifier__learning_rate': 0.5, 'classifier__max_depth': 9, 'classifier__n_estimators': 200}   | Good ROC AUC score, potential overfitting due to high max depth.                                    |
| XGBoost (Random Search)       | 0.9983        | {'classifier__bootstrap': True, 'classifier__max_depth': 20, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 8, 'classifier__n_estimators': 146, 'preprocessor__num__imputer__strategy': 'median'} | Highest ROC AUC score achieved, potentially the best model.                                         |
| XGBoost (Grid Search)         | 0.9971        | {'classifier__learning_rate': 0.1, 'classifier__max_depth': 8, 'classifier__n_estimators': 200}   | Comparable ROC AUC score to Random Forest with Grid Search.                                         |
| LGBM                          | 0.9847        | {'classifier__learning_rate': 0.5, 'classifier__max_depth': 9, 'classifier__n_estimators': 200}   | Good ROC AUC score, potential overfitting due to high max depth.                                    |
| LGBM (Grid Search)            | 0.9946        | {'classifier__learning_rate': 0.5, 'classifier__max_depth': 7, 'classifier__n_estimators': 150}   | Slightly improved ROC AUC score compared to baseline LGBM.                                          |
| CatBoost                      | 0.9781        | {'classifier__learning_rate': 0.5, 'classifier__max_depth': 9, 'classifier__n_estimators': 200}   | Lower ROC AUC score compared to other models.                                                       |
| CatBoost (Grid Search)        | 0.9948        | {'classifier__learning_rate': 0.5, 'classifier__max_depth': 9, 'classifier__n_estimators': 200}   | Slightly improved ROC AUC score compared to CatBoost without Grid Search.                            |
| AdaBoost                      | 0.8673        | None                                                       | Lowest ROC AUC score, potential underfitting or lack of model complexity.                            |
| MLP Classifier                | 0.9934        | None                                                       | Good ROC AUC score, potential overfitting due to lack of regularization.                            |
| SVM (Bayesian Search)         | 0.9575        | {'my_svc__C': 2.3527, 'my_svc__degree': 6, 'my_svc__gamma': 0.0229, 'my_svc__kernel': 'poly'}     | Achieved competitive ROC AUC score compared to other models.                                       |
## Conclusion

Based on the analysis of various models and their performance metrics, it is evident that XGBoost with Random Search (Trial 8) achieved the highest ROC AUC score of 0.9983. This indicates that the XGBoost model trained with randomized hyperparameter search outperforms other models in terms of predicting the outcome of speed dating sessions based on individual profiles.

The Random Forest model with Random Search (Trial 2) and Grid Search (Trial 3) also demonstrated competitive performance, achieving ROC AUC scores of 0.9977 and 0.9978 respectively. These models exhibit robustness and effectiveness in capturing complex relationships within the dataset.

In contrast, models such as AdaBoost and MLP Classifier yielded lower ROC AUC scores, indicating potential limitations in capturing the underlying patterns within the data or susceptibility to overfitting.

Overall, XGBoost with Random Search emerges as the best-performing model in this analysis, providing the highest predictive accuracy for speed dating outcomes. However, further experimentation and optimization may be warranted to explore the full potential of other models and search techniques.

In conclusion, XGBoost with Random Search presents a promising solution for predicting speed dating outcomes, offering valuable insights for matchmaking algorithms and enhancing the overall user experience in speed dating events.
