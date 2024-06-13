# AI-generated-text-detection_using_ML

##**Outcome**

**Observation 1:**

The first observation was that if we used feature section methods like Random Forest classifiers, Principal Components Analysis (PCA), Boruta shadow variables, and K-best feature selection, we get a lower F-1 score. This is because the model cannot capture the entire trend of the dataset and could also be due to loss of information. However, if we retained all the parameters as predictors and run our models, the computational time was quite high. It was an interesting challenge to balance the trade off between interpretability and computational time.

**Observation 2:**

The second observation made was related to oversampling. We chose to not use Synthetic Miniority Over-sampling Technique (SMOTE) to balance the classes in this classification problem. It was initially considered because the target class (1) was not proportional to the other class (0) in the target variable.However, we chose to not proceed with the SMOTE technique because we observed that it created synthetic data which added noise in the model. This reduced the model performance metrics such as the F-1 score, accuracy and precision, ultimately reducing the accurate predictions of the target variable.

**Observation 3:**

The third observation is the role played by word_count in as an important feature in our final model. During the EDA exploration, we found both punc_num and word_count to have a moderately strong positive correlation to the target variable. Upon completing our permutation importance analysis, we found word_count played a key role as the most important contributor in predicting the target variable. Potentially, punc_num does not feature in the list as it has a strong positive correlation with word_count.

**Observation 4:**

The fourth observation was about the model's interpretability. The partial dependece plot for word_count indicates that higher the value of word_count, there is a greater chance of the target variable being 1 i.e. AI-originated. This finding is in contrast to the observations made during EDA (Output 5). The opposite can be noted for feature_512 and featue_386 wherein higher the value of the predictor, higher the chance of the target variable being 0 i.e. human-originated.

**Observation 5:**

The final observation we made was that a single model by itself would not yield a F-1 score higher than 0.68. This was likely due to the fact that the information captured from the complex dataset was limited. Our success lay in the fact that we combined results from multiple models using a stacking ensemble which boosted the accuracy of the target class prediction. The only drawback to our approach is the loss of interpretability that is associated with ensemble models.

##**Appendix - Alternative approaches**
The goal of the project was to optimize the F-1 score and get the best score in the test partition. To achieve this goal, we explored many approaches which are outlined in the Appendix.

Approach 1
Using all the parameters as predictors and building Ensemble Stacking Classifier model gave the results as below.

One alternative approach that we took was using an Ensemble Stacking model. The base models considered were Random Forest Classifier, Support Vector Classifier, Decision Tree Bagging Classifier, and Multilayer Perceptron (MLP) Classifier. We ran the model without using any feature engineering to select important features or reducing the dimensionality. This resulted in us getting an F-1 score of 0.71.


Approach 2
We also explored using the Boruta Shadow variables as part of feature engineering. This helped to reduce the number of parameters that were used as predictors from 770 to 387 predictors. However, with these predictors, XGBoost model gave us the highest F-1 score of 0.64 (64%), which did not perform better relative to the base Logistic Regression model (F-1 score at 0.67).



Approach 3
We also explored randomly selecting 200 parameters with the Stacking Classifier to be used as predictors. However, the F-1 score was found to be 0.67 (67%).



Approach 4
We also explored using a simple MLP classifier model with all the parameters from the dataset as the predictors. Given that all the information from the dataset was at use, the F-1 score improved and was around 0.68 (68%).

