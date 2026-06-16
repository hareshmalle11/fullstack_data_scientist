# K-Nearest Neighbors Regressor - Insurance Charges Prediction

## 1. Model Name

- **Full name:** K-Nearest Neighbors Regressor
- **Short name:** KNN Regressor
- **Category:** Regression
- **Learning type:** Supervised Learning
- **Output type:** Continuous numerical value
- **Project example:** Predicting insurance charges from customer details

---

## 2. Problem It Solves

KNN Regressor solves regression problems where the target output is a number.

In this project, the model predicts:

```text
insurance charges
```

using:

- age
- sex
- BMI
- children
- smoker
- region

### Why We Need This Model

KNN Regressor is useful when similar customers are expected to have similar insurance charges.

For example:

- customers with similar age, BMI, smoker status, and region may have similar charges
- a new customer's charge can be estimated by averaging charges of similar previous customers

KNN is easy to understand and works well as a baseline regression model.

---

## 3. Intuition

### Simple Explanation

KNN Regressor predicts a number by finding the `k` closest training examples and averaging their target values.

Example:

```text
k = 3
nearest charges = 5000, 5500, 6000
prediction = (5000 + 5500 + 6000) / 3
prediction = 5500
```

### Real-World Analogy

Imagine estimating a person's insurance cost by finding similar customers from the past.

If the closest similar customers paid around 5000 to 6000, then the new customer is likely to pay a similar amount.

### Project Context - Insurance Charges Prediction

The local dataset is:

```text
insurance.csv
```

The target column is:

```text
charges
```

The app uses:

| Feature | Meaning |
|---|---|
| `age` | Customer age |
| `sex` | Customer sex |
| `bmi` | Body mass index |
| `children` | Number of children/dependents |
| `smoker` | Whether the customer smokes |
| `region` | Customer region |

The saved model file is:

```text
knn_insurance_model.pkl
```

---

## 4. Mathematical Foundation

### Core Formula - Euclidean Distance

```text
d(A, B) = sqrt((x1_A - x1_B)^2 + (x2_A - x2_B)^2 + ... + (xp_A - xp_B)^2)
```

### KNN Regression Rule

Uniform average:

```text
y_hat = (1/k) * sum(y_i)
```

for the `k` nearest neighbors.

Distance-weighted average:

```text
y_hat = sum(weight_i * y_i) / sum(weight_i)
```

where:

```text
weight_i = 1 / distance_i
```

### Explain Every Term

| Symbol | Meaning |
|---|---|
| `x` | Input feature vector |
| `p` | Number of features |
| `k` | Number of nearest neighbors |
| `d(A, B)` | Distance between two points |
| `y_i` | Target value of neighbor `i` |
| `y_hat` | Predicted numeric value |
| `weight_i` | Influence of neighbor `i` |

---

## 5. Objective Function

KNN Regressor does not minimize a training objective function.

It predicts using local averaging:

```text
Find k nearest training rows and average their target values.
```

### What Is KNN Optimizing?

KNN assumes:

```text
nearby points should have similar target values
```

Good predictions depend on:

- meaningful distance metric
- scaled numerical features
- useful encoded categorical features
- good choice of `k`

---

## 6. Loss Function Derivation

KNN does not learn weights using a loss function.

There is no gradient descent and no coefficient update.

For prediction:

1. Calculate distance from new input to every training row.
2. Sort rows by distance.
3. Select the closest `k`.
4. Average their `charges`.

### Evaluation Loss

After predictions, regression loss can be measured using:

```text
MAE, MSE, RMSE, R2
```

These metrics evaluate quality, but they are not used to train KNN directly.

---

## 7. Optimization Method

KNN is a lazy learning method.

### Training

```text
store training data
```

### Prediction

```text
compute distances
find nearest neighbors
average target values
```

### Neighbor Search Options

- brute force
- KD-tree
- Ball tree

For small datasets, brute force is usually fine.

For larger datasets, tree-based search can speed up prediction.

---

## 8. Training Workflow

```text
Input Data
down arrow
Handle Missing Values
down arrow
Encode Categorical Features
down arrow
Scale Features
down arrow
Split Train/Test Data
down arrow
Store Training Rows
down arrow
Receive New Customer Input
down arrow
Compute Distances
down arrow
Find k Nearest Customers
down arrow
Average Their Charges
down arrow
Predict Insurance Charge
down arrow
Evaluate Regression Metrics
down arrow
Save Model
```

For this project:

1. Load `insurance.csv`.
2. Use `charges` as target.
3. Encode `sex`, `smoker`, and `region`.
4. Use age, BMI, children, and encoded features for distance.
5. Train/store KNN Regressor.
6. Save model as `knn_insurance_model.pkl`.
7. Use `App.py` for prediction.

---

## 9. Worked Numerical Example

Suppose we predict charges using two features:

```text
age
bmi
```

Training data:

| Customer | Age | BMI | Charges |
|---|---:|---:|---:|
| A | 25 | 26 | 4000 |
| B | 30 | 28 | 5000 |
| C | 45 | 35 | 12000 |
| D | 50 | 37 | 15000 |

New customer:

```text
age = 29
bmi = 27
```

Distance to A:

```text
sqrt((29 - 25)^2 + (27 - 26)^2)
= sqrt(16 + 1)
= 4.12
```

Distance to B:

```text
sqrt((29 - 30)^2 + (27 - 28)^2)
= sqrt(1 + 1)
= 1.41
```

Distance to C:

```text
sqrt((29 - 45)^2 + (27 - 35)^2)
= sqrt(256 + 64)
= 17.89
```

Distance to D:

```text
sqrt((29 - 50)^2 + (27 - 37)^2)
= sqrt(441 + 100)
= 23.26
```

If `k = 2`, nearest neighbors are:

```text
B and A
```

Predicted charges:

```text
(5000 + 4000) / 2 = 4500
```

Final prediction:

```text
4500
```

---

## 10. Full Manual Training Example

KNN does not train coefficients, so the manual example shows stored data and prediction.

### Stored Training Rows

| Row | Age | BMI | Smoker Encoded | Charges |
|---|---:|---:|---:|---:|
| 1 | 22 | 25 | 0 | 2500 |
| 2 | 30 | 28 | 0 | 4200 |
| 3 | 45 | 34 | 1 | 18000 |
| 4 | 50 | 36 | 1 | 22000 |
| 5 | 29 | 27 | 0 | 4500 |

### New Customer

```text
age = 31
bmi = 29
smoker = 0
```

### Distances

| Row | Distance Summary | Distance | Charges |
|---|---|---:|---:|
| 1 | farther age/bmi | 9.85 | 2500 |
| 2 | very close | 1.41 | 4200 |
| 3 | smoker difference and far | 17.49 | 18000 |
| 4 | smoker difference and far | 22.05 | 22000 |
| 5 | close | 2.83 | 4500 |

### Select `k = 3`

Nearest rows:

```text
2, 5, 1
```

Their charges:

```text
4200, 4500, 2500
```

### Average

```text
(4200 + 4500 + 2500) / 3
= 11200 / 3
= 3733.33
```

Final prediction:

```text
3733.33
```

---

## 11. Parameters - Learned Values

KNN has no learned coefficients.

| Stored Item | Meaning |
|---|---|
| Training features | Stored customer examples |
| Training targets | Insurance charges for those customers |
| Search structure | Optional neighbor-search index |

---

## 12. Hyperparameters

| Hyperparameter | Purpose |
|---|---|
| `n_neighbors` | Number of neighbors |
| `weights` | Uniform or distance-weighted average |
| `metric` | Distance metric |
| `p` | Minkowski distance power |
| `algorithm` | Neighbor search strategy |
| `leaf_size` | Tree search tuning |

---

## 13. Why Each Hyperparameter Exists

### n_neighbors

Small `k`:

- low bias
- high variance
- can overfit

Large `k`:

- smoother predictions
- lower variance
- can underfit

### weights

Uniform:

- all neighbors contribute equally

Distance:

- closer neighbors contribute more

### metric

The distance metric defines what "similar customer" means.

---

## 14. Assumptions

- Similar customers have similar charges.
- Distance between rows is meaningful.
- Features are scaled.
- Categorical features are encoded carefully.
- Training data represents future users.

---

## 15. Data Requirements

| Requirement | Needed? | Why |
|---|---|---|
| Numeric target | Yes | Regression predicts numbers |
| Feature scaling | Yes | Distance is scale sensitive |
| Categorical encoding | Yes | KNN needs numeric features |
| Missing value handling | Yes | Missing values break distance calculation |
| Outlier checking | Recommended | Outliers can affect local averages |

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

### Prediction

```text
O(n * p)
```

for brute force distance calculation.

### Memory

```text
O(n * p)
```

because KNN stores training data.

---

## 17. Decision Boundary

Regression models do not have class decision boundaries.

KNN Regressor creates local prediction regions.

Example:

```text
charges
  ^
  |              high-charge smoker region
  |        ________
  |_______|
  | low-charge non-smoker region
  +--------------------------------> age / bmi
```

### Linear or Nonlinear?

KNN Regressor is nonlinear because predictions depend on local neighbors.

---

## 18. Overfitting and Underfitting

### Overfitting

Happens when `k` is too small.

With `k = 1`, the prediction exactly follows the nearest training target, including noise.

### Underfitting

Happens when `k` is too large.

The prediction becomes a broad average and may ignore local differences.

---

## 19. Regularization

KNN does not use L1, L2, or Elastic Net regularization.

Model smoothness is controlled by:

- `n_neighbors`
- distance weighting
- feature selection
- scaling

### L1

Not directly used.

### L2

Euclidean distance is L2 distance, but this is not L2 regularization.

### Elastic Net

Not applicable.

---

## 20. Feature Importance

KNN does not provide built-in coefficients.

Feature importance can be estimated using:

- permutation importance
- removing one feature at a time
- comparing validation error

For this insurance project, important features may include:

- smoker
- age
- BMI
- children
- region
- sex

---

## 21. Advantages

- Very easy to understand.
- No complex training step.
- Can model nonlinear patterns.
- Works for regression and classification.
- Useful baseline model.

---

## 22. Limitations

- Slow prediction for large datasets.
- Sensitive to scaling.
- Sensitive to irrelevant features.
- Sensitive to outliers.
- Needs storing training data.
- Struggles in high-dimensional data.

---

## 23. Failure Cases

KNN Regressor can fail when:

- features are not scaled
- encoded categories distort distance
- many irrelevant features exist
- dataset is very large
- target has extreme outliers
- similar customers do not actually have similar charges

---

## 24. Edge Cases

### Dataset Is Tiny

Predictions may be unstable.

### Dataset Is Huge

Prediction can become slow.

### Features More Than Samples

Distance becomes less reliable.

### Imbalanced Target

If most charges are low and few are very high, KNN may underpredict rare expensive cases.

### Missing Values

Missing values must be imputed before prediction.

---

## 25. Evaluation Metrics

### MAE

```text
MAE = (1/n) * sum(|y_i - y_hat_i|)
```

### MSE

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

### RMSE

```text
RMSE = sqrt(MSE)
```

### R2 Score

```text
R2 = 1 - (SS_res / SS_tot)
```

Classification metrics such as accuracy and F1 are not used for this regression task.

---

## 26. Comparison with Similar Models

| Feature | KNN Regressor | Decision Tree Regressor | Linear Regression |
|---|---|---|---|
| Training speed | Very fast | Fast | Fast |
| Prediction speed | Slow for large data | Fast | Fast |
| Scaling needed | Yes | No | Recommended |
| Nonlinear patterns | Yes | Yes | No, unless engineered |
| Interpretability | Medium | High | High |
| Large data handling | Weak | Better | Good |

---

## 27. Real-World Applications

- Insurance charge prediction
- House price prediction
- Recommendation systems
- Medical cost estimation
- Similar-customer pricing
- Demand estimation
- Local interpolation problems

---

## 28. Scikit-Learn Implementation

```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### With Scaling and Encoding

```python
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("knn", KNeighborsRegressor(n_neighbors=5)),
    ]
)

model.fit(X_train, y_train)
```

---

## 29. Interview Questions

1. What is KNN Regressor?
2. How is KNN Regressor different from KNN Classifier?
3. Why is KNN called lazy learning?
4. What does `k` control?
5. How does KNN predict a continuous value?
6. Why is feature scaling important?
7. What happens if `k` is too small?
8. What happens if `k` is too large?
9. What distance metrics can be used?
10. What is distance-weighted prediction?
11. Why is KNN slow for large datasets?
12. Which regression metrics are used?
13. How do outliers affect KNN regression?
14. What is the target column in this insurance project?
15. Which features are used in this project?

---

## 30. Summary

KNN Regressor is a simple supervised regression algorithm.

It predicts a number by finding nearby training examples and averaging their target values.

It does not learn coefficients during training. It stores the training data and uses distance calculations at prediction time.

In this project, KNN predicts insurance charges using customer features such as age, BMI, smoker status, children, sex, and region.
