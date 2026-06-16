# K-Nearest Neighbors Classifier - Rain Prediction

## 1. Model Name

- **Full name:** K-Nearest Neighbors Classifier
- **Short name:** KNN Classifier
- **Category:** Classification
- **Learning type:** Supervised Learning
- **Output type:** Class label
- **Project example:** Predicting rain or no rain from weather features

---

## 2. Problem It Solves

KNN Classifier solves classification problems where the target output is a category.

In this project, the model predicts:

```text
rain
no rain
```

using weather features:

- Temperature
- Humidity
- Wind Speed
- Cloud Cover
- Pressure

### Why We Need This Model

KNN is useful when similar inputs are expected to have similar outputs.

For rain prediction:

- a day with high humidity and high cloud cover may be similar to previous rainy days
- a day with low humidity and low cloud cover may be similar to previous no-rain days

KNN does not learn complex equations during training. It stores the training data and classifies new points by looking at nearby examples.

---

## 3. Intuition

### Simple Explanation

KNN predicts a new data point by checking the `k` closest training examples.

For classification, the final class is decided by majority vote.

Example:

```text
k = 5
nearest neighbors = rain, rain, no rain, rain, no rain
majority = rain
prediction = rain
```

### Real-World Analogy

Imagine moving to a new city and asking your nearest neighbors whether it usually rains in this weather condition.

If most nearby examples say it rains, you predict rain.

If most nearby examples say no rain, you predict no rain.

KNN works exactly like this: it asks nearby examples.

### Project Context - Rain Prediction

The local dataset is:

```text
weather_forecast_data.csv
```

The target column is:

```text
Rain
```

The Streamlit app uses:

| Feature | Meaning |
|---|---|
| `Temperature` | Air temperature |
| `Humidity` | Moisture level in air |
| `Wind_Speed` | Wind speed |
| `Cloud_Cover` | Cloud coverage |
| `Pressure` | Atmospheric pressure |

The saved model file is:

```text
knn_rain_model.pkl
```

---

## 4. Mathematical Foundation

### Core Formula - Euclidean Distance

For two points `A` and `B`:

```text
d(A, B) = sqrt((x1_A - x1_B)^2 + (x2_A - x2_B)^2 + ... + (xp_A - xp_B)^2)
```

### KNN Classification Rule

```text
y_hat = majority_class among k nearest neighbors
```

### Explain Every Term

| Symbol | Meaning |
|---|---|
| `x` | Input feature vector |
| `p` | Number of features |
| `k` | Number of nearest neighbors |
| `d(A, B)` | Distance between two points |
| `y_hat` | Predicted class |
| `xj_A` | Feature `j` of point `A` |
| `xj_B` | Feature `j` of point `B` |

### Other Distance Metrics

Manhattan distance:

```text
d(A, B) = sum(|xj_A - xj_B|)
```

Minkowski distance:

```text
d(A, B) = (sum(|xj_A - xj_B|^p))^(1/p)
```

Cosine distance can be useful for text-like high-dimensional data.

---

## 5. Objective Function

KNN does not train by minimizing a traditional objective function like linear regression or neural networks.

Instead, prediction is based on local similarity:

```text
Find the k closest training examples and choose the most common class.
```

### What Is KNN Optimizing?

KNN implicitly assumes:

```text
nearby points should have similar labels
```

The quality of KNN depends on:

- distance metric
- feature scaling
- value of `k`
- quality of training data

### What Happens When Distance Is Small?

Small distance means the new point is similar to a training point.

That neighbor has strong influence.

### What Happens When Distance Is Large?

Large distance means the training point is less similar.

It should have little or no influence.

---

## 6. Loss Function Derivation

KNN has no parameter-learning loss function.

There are no learned weights like:

```text
w1, w2, b
```

Instead, KNN follows this prediction process:

1. Store all training rows.
2. For a new row, calculate distance to every training row.
3. Sort distances.
4. Select the closest `k` rows.
5. Count class labels.
6. Return the majority class.

### Intuition

The "loss" appears during evaluation, not training.

After predictions are made, we measure classification errors using:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

### Gradient

KNN does not use gradient descent.

There is no gradient update because KNN does not learn coefficients.

---

## 7. Optimization Method

KNN is a **lazy learning** algorithm.

### Lazy Learning Meaning

Training phase:

```text
store the dataset
```

Prediction phase:

```text
perform distance calculations
```

KNN delays most computation until prediction time.

### Search Optimization

For faster neighbor search, implementations may use:

- brute force search
- KD-tree
- Ball tree

### Voting Rule

Uniform voting:

```text
each neighbor gets one vote
```

Distance-weighted voting:

```text
closer neighbors get stronger votes
```

---

## 8. Training Workflow

```text
Input Data
down arrow
Clean Missing Values
down arrow
Scale Numerical Features
down arrow
Split Train/Test Data
down arrow
Store Training Data
down arrow
Receive New Input
down arrow
Compute Distances
down arrow
Find k Nearest Neighbors
down arrow
Majority Vote
down arrow
Predict Class
down arrow
Evaluate Classification Metrics
down arrow
Save Model
```

For this project:

1. Load `weather_forecast_data.csv`.
2. Use weather columns as features.
3. Use `Rain` as the target.
4. Scale features if the pipeline includes scaling.
5. Train/store KNN classifier.
6. Save model as `knn_rain_model.pkl`.
7. Use `App.py` for prediction.

---

## 9. Worked Numerical Example

Suppose we predict rain using two features:

```text
Humidity
Cloud Cover
```

Training data:

| Day | Humidity | Cloud Cover | Class |
|---|---:|---:|---|
| A | 80 | 70 | rain |
| B | 75 | 65 | rain |
| C | 40 | 20 | no rain |
| D | 45 | 25 | no rain |

New day:

```text
Humidity = 78
Cloud Cover = 68
```

Use Euclidean distance.

Distance to A:

```text
sqrt((78 - 80)^2 + (68 - 70)^2)
= sqrt(4 + 4)
= sqrt(8)
= 2.83
```

Distance to B:

```text
sqrt((78 - 75)^2 + (68 - 65)^2)
= sqrt(9 + 9)
= sqrt(18)
= 4.24
```

Distance to C:

```text
sqrt((78 - 40)^2 + (68 - 20)^2)
= sqrt(1444 + 2304)
= sqrt(3748)
= 61.22
```

Distance to D:

```text
sqrt((78 - 45)^2 + (68 - 25)^2)
= sqrt(1089 + 1849)
= sqrt(2938)
= 54.20
```

If `k = 3`, nearest neighbors are:

```text
A = rain
B = rain
D = no rain
```

Majority vote:

```text
rain = 2
no rain = 1
```

Final prediction:

```text
rain
```

---

## 10. Full Manual Training Example

KNN does not train weights, so "manual training" means storing rows and manually predicting.

### Initial Stored Data

| Row | Temperature | Humidity | Cloud Cover | Class |
|---|---:|---:|---:|---|
| 1 | 24 | 85 | 80 | rain |
| 2 | 28 | 45 | 10 | no rain |
| 3 | 25 | 82 | 60 | rain |
| 4 | 30 | 50 | 20 | no rain |
| 5 | 23 | 75 | 70 | rain |

### New Input

```text
Temperature = 26
Humidity = 80
Cloud Cover = 65
```

### Distances

Using three features:

| Row | Calculation Summary | Distance | Class |
|---|---|---:|---|
| 1 | close in all features | 7.35 | rain |
| 2 | far in humidity/cloud | 72.84 | no rain |
| 3 | very close | 5.48 | rain |
| 4 | far in humidity/cloud | 73.41 | no rain |
| 5 | close | 7.07 | rain |

### Select `k = 3`

Nearest rows:

```text
3, 5, 1
```

Their classes:

```text
rain, rain, rain
```

### Final Vote

```text
rain = 3
no rain = 0
```

Final prediction:

```text
rain
```

---

## 11. Parameters - Learned Values

KNN has no learned coefficients.

| Stored Item | Meaning |
|---|---|
| Training feature rows | Stored reference points |
| Training labels | Class labels used for voting |
| Distance structure | Optional KD-tree/Ball tree for faster search |

KNN is called non-parametric because the model complexity grows with the dataset size.

---

## 12. Hyperparameters

| Hyperparameter | Purpose |
|---|---|
| `n_neighbors` | Number of neighbors `k` |
| `weights` | Uniform or distance-weighted voting |
| `metric` | Distance metric |
| `p` | Power parameter for Minkowski distance |
| `algorithm` | Neighbor search method |
| `leaf_size` | Tree search tuning parameter |

---

## 13. Why Each Hyperparameter Exists

### n_neighbors

Small `k`:

- flexible boundary
- can overfit noise

Large `k`:

- smoother boundary
- can underfit

### weights

Uniform:

- every neighbor has equal vote

Distance:

- closer neighbors vote more strongly

### metric

Different distance metrics define similarity differently.

Choosing the right metric is important because KNN depends fully on distance.

---

## 14. Assumptions

- Similar points have similar labels.
- Features are meaningful for distance calculation.
- Data is scaled properly.
- Training data represents future data.
- Noise and outliers are controlled.

---

## 15. Data Requirements

| Requirement | Needed? | Why |
|---|---|---|
| Feature scaling | Yes | Distance-based models are scale sensitive |
| Missing value handling | Yes | KNN cannot use missing values directly |
| Encoding | Yes, if categorical | Distance needs numeric values |
| Enough training data | Yes | Predictions depend on neighbors |
| Outlier checking | Recommended | Outliers can affect nearest neighbors |

---

## 16. Complexity

Let:

- `n` = training rows
- `p` = features
- `k` = neighbors

### Training

```text
O(1) or O(n)
```

KNN mostly stores data.

### Prediction

Brute force for one point:

```text
O(n * p)
```

Sorting neighbors:

```text
O(n log n)
```

### Memory

```text
O(n * p)
```

KNN stores the training dataset.

---

## 17. Decision Boundary

KNN can create nonlinear decision boundaries.

Example:

```text
Humidity
  ^
  | rain region      rain region
  |     * * *      *
  |       * * *  *
  |--------------------------> Cloud Cover
  | no rain region
```

### Linear or Nonlinear?

Usually nonlinear.

Small `k` gives more jagged boundaries.

Large `k` gives smoother boundaries.

---

## 18. Overfitting and Underfitting

### Overfitting

Happens when `k` is too small.

Example:

```text
k = 1
```

The model follows noise too closely.

### Underfitting

Happens when `k` is too large.

The model becomes too smooth and may ignore local patterns.

---

## 19. Regularization

KNN does not use L1 or L2 regularization.

Complexity is controlled by:

- `n_neighbors`
- distance metric
- feature scaling
- distance weighting

### L1

Not directly used.

### L2

Euclidean distance is related to L2 distance, but that is not L2 regularization.

### Elastic Net

Not applicable.

---

## 20. Feature Importance

KNN does not naturally provide coefficients or feature importances.

Feature influence can be studied using:

- permutation importance
- feature removal experiments
- distance sensitivity analysis

For this rain project, likely useful features are:

- humidity
- cloud cover
- pressure
- temperature
- wind speed

---

## 21. Advantages

- Very simple to understand.
- No training equation required.
- Works for nonlinear boundaries.
- Naturally supports multi-class classification.
- Useful baseline model.

---

## 22. Limitations

- Slow prediction on large datasets.
- Sensitive to feature scaling.
- Sensitive to irrelevant features.
- Sensitive to outliers.
- Requires storing training data.
- Performs poorly in very high dimensions.

---

## 23. Failure Cases

KNN can fail when:

- features are not scaled
- irrelevant features dominate distance
- dataset is very large
- classes overlap heavily
- data is noisy
- useful similarity cannot be captured by the chosen distance metric

---

## 24. Edge Cases

### Dataset Is Tiny

The model may be unstable because there are too few neighbors.

### Dataset Is Huge

Prediction can be slow.

### Features More Than Samples

Distance becomes less meaningful.

### Imbalanced Classes

Majority class may dominate neighbor voting.

### Missing Values

Missing values must be imputed before prediction.

---

## 25. Evaluation Metrics

### Accuracy

```text
Accuracy = correct predictions / total predictions
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1 Score

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Confusion Matrix

Shows correct and incorrect predictions for each class.

---

## 26. Comparison with Similar Models

| Feature | KNN Classifier | Decision Tree | SVM |
|---|---|---|---|
| Training speed | Very fast | Fast | Medium/slow |
| Prediction speed | Slow for large data | Fast | Medium |
| Scaling needed | Yes | No | Yes |
| Nonlinear boundary | Yes | Yes | Yes with kernel |
| Interpretability | Medium | High | Medium |
| Handles large data | Poorly | Better | Depends |

---

## 27. Real-World Applications

- Rain prediction
- Recommendation systems
- Medical diagnosis
- Handwriting recognition
- Customer segmentation labels
- Fraud detection
- Similar-case retrieval

---

## 28. Scikit-Learn Implementation

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### With Scaling

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

model.fit(X_train, y_train)
```

---

## 29. Interview Questions

1. What is KNN?
2. Why is KNN called a lazy learner?
3. What does `k` mean?
4. How does KNN classify a new point?
5. Why is feature scaling important?
6. What happens when `k = 1`?
7. What happens when `k` is very large?
8. What distance metrics can KNN use?
9. How does distance-weighted voting work?
10. Why is KNN slow during prediction?
11. How does KNN handle nonlinear boundaries?
12. What metrics are used for classification?
13. How does class imbalance affect KNN?
14. What is the target column in this rain project?
15. Which weather features are used in this project?

---

## 30. Summary

KNN Classifier is a simple supervised classification algorithm.

It predicts a class by finding the nearest training examples and using majority vote.

It does not learn weights during training. It stores the dataset and performs most work during prediction.

In this project, KNN predicts rain or no rain using temperature, humidity, wind speed, cloud cover, and pressure.
