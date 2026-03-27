# Decision log

## Decision 1: Data cleaning strategy
**What I did:**
I implemented a custom date parser to extract 'year', 'month', and 'day' from inconsistent formats as that can be one of the features on which target output column could be dependent as different months, years can have many admitted patients due to atmosphere. I capped 'age' at 100 years, treating values like 999 as outliers, and used median imputation for missing glucose levels and age. Finally, I applied a log transformation to skewed features like length_of_stay_days and creatinine_mgdl.

**Why I did it:**
The raw data contained physiological impossibilities (age 999) and mixed date strings that would break standard models. Glucose levels were missing in ~15% of records; median imputation was chosen over dropping rows as there were >5% missing values. Log transformation was critical because length_of_stay showed a skewness of 2.12, which often destabilizes neural network gradients.

**What I considered and rejected:**
I considered to charlson_comorbidity_index and prior_admissions_1yr merge into health_risk_score as they were highly correlated features but I rejected that as it was not affecting anything.

**What would happen if I was wrong here:**
The model would not predict correct readmitted in 30 days or not.

---

## Decision 2: Model architecture and handling class imbalance
**What I did:**
I built a 4-layer Sequential Neural Network (64-32-16-1 nodes) using ReLU activations and a Sigmoid output. To handle the 9:1 class imbalance, I calculated and applied class_weights (approx. 5.5 for the minority class) during the .fit() process and used Batch Normalization/Dropout (0.3) for regularization.

**Why I did it:**
With only 342 positive readmission cases out of 3,800, a standard model would simply predict "0" for everyone to achieve 91% accuracy. Class weighting forces the loss function to penalize misclassifying a readmitted patient more heavily.

**What I considered and rejected:**
I considered FOCAL loss to implement and I implemented it too but I did not go further with that as the dataset is small and also model was not able to learn.

**What would happen if I was wrong here:**
The model would achieve high recall for majority class but low for minority.

---

## Decision 3: Evaluation metric and threshold selection
**What I did:**
I moved away from Accuracy and prioritized the F1-Score and Recall for the minority class. After training, I performed a threshold scan (0.1 to 0.6) on the validation set to find the optimal cutoff, settling on approximately 0.57.

**Why I did it:**
In a readmission context, a False Negative (missing a patient who will return) is more "expensive" than a False Positive. However, to maintain a balance where the model is still precise enough to be actionable, the F1-score optimization provided the best middle ground. The default 0.5 threshold is rarely optimal for weighted-loss models.

**What I considered and rejected:**
I rejected using Accuracy as for the medical data Recall is more important than Acurracy.

**What would happen if I was wrong here:**
We might miss a significant portion of at-risk patients that the model was actually capable of identifying with a slightly adjusted boundary.