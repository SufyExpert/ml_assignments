# Observations — Stroke Prediction Benchmark

## What I observed

I ran the benchmark on the stroke dataset and compared the outputs across the models. The most important observation is that accuracy looks high for most models, but it is not a reliable indicator in this problem because the dataset is imbalanced. A model can achieve around 95% accuracy while still failing to identify the actual stroke cases. That means the benchmark must be judged mainly by recall, precision, and class-level behavior, not only by overall accuracy.

The strongest evidence from the results is that the positive class was very weakly detected. In the test results, many models had precision of 0.0, recall of 0.0, or F1 close to 0.0. This shows that the models were mostly predicting the majority class and not learning the stroke pattern effectively.

## Basic findings from the results

I observed the following points from the actual test results:

- Logistic Regression reached an accuracy of 0.952, but its recall was only 0.02 and F1 was 0.039. Its precision was 1.0, which means it was very conservative and only predicted the positive class in a very small number of cases.
- Gradient Boosting reached 0.950 accuracy, but recall stayed at 0.02 and F1 at 0.038. This indicates that the model ranked the classes somewhat well, but the decision threshold still caused poor positive detection.
- KNN models delivered accuracy around 0.946–0.951, but their recall and F1 were 0.0 in test performance. This suggests poor separation of the minority class.
- Random Forest and XGBoost also remained weak on the stroke class, with accuracy near 0.945 but precision and recall close to 0.0.
- The deep Decision Tree had a test accuracy of 0.907, but its precision, recall, and F1 were still low. This is a sign of overfitting and weak generalization for the minority class.

## Strong reasoning behind these results

The main reason is class imbalance. The positive class is much smaller than the negative class, so a model can gain high accuracy simply by predicting the non-stroke class most of the time. In this dataset, that problem is serious because the score is not only looking at a global accuracy value, but at whether the model can actually catch the patients with stroke.

The second reason is the default decision threshold. Most models were evaluated using a 0.5 cutoff. If the predicted probabilities for stroke cases are low, the model will not classify them as positive even when its ranking ability is not terrible. This is why ROC-AUC can look moderately acceptable while recall remains poor.

The third reason is overfitting. The deep Decision Tree showed high training performance and then weaker validation/test performance. This tells me the model learned the training data too specifically and could not generalize well. The same pattern is visible in other tree-based models, where the training patterns look stronger than the actual test behavior.

## Questions we covered:

### 1. Which model performed best on training data?

- I observed that the best-performing models on the training set were the deep Decision Tree and Random Forest. Both reached near-perfect training scores, with accuracy, precision, recall, F1, and ROC-AUC all close to 1.0. This means they fit the training data almost perfectly.

- However, this level of training performance is also a warning sign. It indicates that these models learned the training patterns too specifically and are likely overfitting. In other words, they memorized the data rather than learning a boundary that would generalize well.

### 2. Which model performed best on validation data?

- On the validation set, Logistic Regression was the strongest performer. Its validation ROC-AUC was around 0.841, and its validation accuracy stayed close to 0.951. This makes it the most stable model during validation.

- Compared with the other models, Logistic Regression showed better balance between class separation and generalization. The more complex models either overfit the training data or failed to identify the minority class in a useful way.

### 3. Which model generalized best to the final test set?

- I concluded that Logistic Regression generalized best to the final test set. It had the highest test ROC-AUC, approximately 0.843, and maintained a strong accuracy level around 0.952. This was the most reliable result among the models I tested.

- Even so, the final test results still showed a clear limitation. Many models had recall values near 0.0 and precision values of 0.0 or very low. This means that although overall accuracy looked strong, the models were still not detecting the stroke cases effectively enough.

### 4. Which algorithm was most interpretable?

- The most interpretable algorithm was the Decision Tree. It is easy to explain because its predictions are made through a sequence of simple if-then rules. A shallow Decision Tree is especially interpretable because it is shorter and easier for a person to follow.

- Between the shallow and deep tree versions, the shallow tree was clearly more interpretable. The deep tree may fit the training data better, but it becomes harder to understand and explain.

### 5. Which algorithm was fastest at inference time?

- The fastest model at inference time was the deep Decision Tree, followed closely by Logistic Regression. In the final output, the deep Decision Tree had the lowest inference time on the test set.

- This difference is small, so I would not treat speed as the main deciding factor. Predictive quality and class detection are more important than small timing differences in this benchmark.

### 6. Which model would you choose if explainability were a requirement?

- If explainability were the main requirement, I would choose the Decision Tree. It is the easiest model to explain because the decision process is transparent and can be followed directly.

- However, the trade-off is clear. The shallow tree was easy to explain, but it was also too simple to capture the pattern properly. So the more interpretable model is not always the best predictive model.

### 7. Which model would you choose if predictive performance were the primary objective?

- If predictive performance were the main objective, I would choose Logistic Regression as the best overall model in this benchmark. It showed the strongest and most stable generalization behavior on the validation and test sets, with the highest ROC-AUC values among the models.

- At the same time, I would still be cautious because its recall for the positive class remained low. This means it is not yet a strong stroke-detection model in a real-world clinical context without further tuning and imbalance handling.

### 8. Did any model show signs of high bias or high variance?

- Yes. I observed both patterns clearly:

   - **High bias:** The shallow Decision Tree and larger KNN models showed signs of underfitting. They were too simple and unable to capture enough of the underlying structure.
   - **High variance:** The deep Decision Tree and Random Forest showed clear signs of overfitting. They reached near-perfect training scores but performed worse on validation and test data.

- This contrast is useful because it shows that the issue is not only the choice of algorithm. The dataset is difficult, the class distribution is uneven, and the models struggle to learn the stroke-positive class consistently.

## What I would conclude

From the results, I conclude that the benchmark worked as intended, but the dataset structure makes the task difficult. The models are not completely failing because they are badly built; they are failing because the positive class is too small and the evaluation is too dependent on accuracy. For this type of medical classification task, the important question is not whether the overall accuracy is high, but whether the model can detect the actual stroke patients.

If this project were improved further, I would focus on class weighting, threshold tuning, and stronger handling of imbalance before making final conclusions about model quality. Without that, many models will look good in summary but still fail in the most important part of the task.

## Final takeaway

My main conclusion is that the project is successful in showing the benchmark process, but the real challenge is not just model selection. The main issue is imbalance and decision thresholding. The results show that the dataset is harder than it appears from accuracy alone, and the benchmark demonstrates that the chosen models do not reliably detect the stroke-positive class under the current setup.
