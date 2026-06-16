# Ridge Regression and Lasso Regression - Car Price Prediction

## 1. Model Name

- **Full names:** Ridge Regression and Lasso Regression
- **Category:** Regression
- **Learning type:** Supervised Learning
- **Output type:** Continuous numerical value
- **Example project:** Predicting car price from engine, body, mileage, and categorical car specifications

---

## 2. Problem It Solves

Ridge and Lasso Regression solve regression problems where the goal is to predict a continuous number.

In this project, the model predicts **car price** using features such as:

- fuel type
- car body
- engine size
- horsepower
- curb weight
- mileage
- wheelbase
- engine type
- number of cylinders

### Why We Need These Models

Simple Linear Regression can overfit when:

- there are many input features
- features are correlated with each other
- the dataset is small
- the model learns very large coefficients

Ridge and Lasso improve Linear Regression by adding **regularization**, which penalizes overly large weights.

- **Ridge Regression** uses L2 regularization.
- **Lasso Regression** uses L1 regularization.

Regularization helps the model generalize better on unseen data.

---

## 3. Intuition

### Simple Explanation

Linear Regression finds weights for each feature so that predictions are close to the actual values.

Ridge and Lasso do the same thing, but they add one extra rule:

> Do not allow the weights to become unnecessarily large.

Large weights usually mean the model is depending too heavily on specific features. That can make the model memorize training data instead of learning a stable pattern.

### Ridge Intuition

Ridge shrinks weights toward zero, but usually does not make them exactly zero.

This means Ridge keeps most features but reduces their influence.

### Lasso Intuition

Lasso can shrink some weights exactly to zero.

This means Lasso can automatically perform feature selection by removing less useful features.

### Real-World Analogy

Imagine a teacher grading students.

Simple Linear Regression may give too much importance to one factor, such as homework score.

Ridge says:

> Use all factors, but do not let any one factor dominate too much.

Lasso says:

> If a factor is not useful, ignore it completely.

For car price prediction:

- Ridge may keep horsepower, engine size, mileage, fuel type, and body type.
- Lasso may decide some weak features are not important and assign them weight 0.

---

## 4. Mathematical Foundation

### Linear Regression Base Formula

For one input feature:

```text
y_hat = b + w x
```

For multiple input features:

```text
y_hat = b + w1*x1 + w2*x2 + ... + wp*xp
```

Vector form:

```text
y_hat = Xw + b
```

### Ridge Regression Objective

```text
Ridge Loss = MSE + lambda * sum(w_j^2)
```

or

```text
J(w) = (1/n) * sum((y_i - y_hat_i)^2) + lambda * sum(w_j^2)
```

### Lasso Regression Objective

```text
Lasso Loss = MSE + lambda * sum(|w_j|)
```

or

```text
J(w) = (1/n) * sum((y_i - y_hat_i)^2) + lambda * sum(|w_j|)
```

### Explain Every Term

| Symbol | Meaning |
|---|---|
| `y` | Actual target value |
| `y_hat` | Predicted target value |
| `x_j` | Input feature number `j` |
| `w_j` | Weight/coefficient for feature `j` |
| `b` | Bias/intercept |
| `n` | Number of training rows |
| `p` | Number of features |
| `lambda` | Regularization strength |
| `sum(w_j^2)` | L2 penalty used by Ridge |
| `sum(|w_j|)` | L1 penalty used by Lasso |

### Important Note About `lambda` and `alpha`

In many books, regularization strength is written as `lambda`.

In scikit-learn, it is called `alpha`.

So:

```text
lambda = alpha
```

---

## 5. Objective Function

The model tries to minimize prediction error while also keeping weights controlled.

### Linear Regression Objective

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

Linear Regression only minimizes error.

### Ridge Objective

```text
MSE + L2 penalty
```

Ridge minimizes:

- prediction error
- squared size of weights

### Lasso Objective

```text
MSE + L1 penalty
```

Lasso minimizes:

- prediction error
- absolute size of weights

### Why This Objective?

The MSE term makes predictions accurate.

The regularization term prevents the model from becoming too complex.

### What Happens When Objective Increases?

If the objective increases, it means:

- predictions are becoming worse, or
- weights are becoming too large, or
- both are happening

### What Happens When Objective Decreases?

If the objective decreases, it means:

- predictions are improving, or
- weights are becoming simpler, or
- both are improving

---

## 6. Loss Function Derivation

### Step 1: Prediction

For a single row:

```text
y_hat_i = b + w1*x_i1 + w2*x_i2 + ... + wp*x_ip
```

### Step 2: Error

```text
error_i = y_i - y_hat_i
```

### Step 3: Squared Error

```text
error_i^2 = (y_i - y_hat_i)^2
```

### Step 4: Mean Squared Error

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

### Step 5: Add Regularization

Ridge:

```text
J(w) = MSE + lambda * sum(w_j^2)
```

Lasso:

```text
J(w) = MSE + lambda * sum(|w_j|)
```

### Gradient Intuition

A gradient tells the model:

- which direction to move weights
- how strongly to move them

For MSE:

```text
dMSE/dw_j = (-2/n) * sum(x_ij * (y_i - y_hat_i))
```

For Ridge:

```text
dJ/dw_j = dMSE/dw_j + 2 * lambda * w_j
```

For Lasso:

```text
dJ/dw_j = dMSE/dw_j + lambda * sign(w_j)
```

Lasso is not differentiable exactly at `w_j = 0`, so optimization methods use subgradients or coordinate descent.

---

## 7. Optimization Method

### Ridge Regression

Ridge can be optimized using:

- closed-form solution
- gradient descent
- solvers such as `auto`, `svd`, `cholesky`, `lsqr`, or `sag`

### Ridge Closed-Form Solution

```text
w = (X^T X + lambda I)^(-1) X^T y
```

Where:

- `X^T` is the transpose of the feature matrix
- `I` is the identity matrix
- `lambda I` adds stability and penalizes large weights

The bias term is usually not regularized.

### Ridge Gradient Descent Update

```text
w_j = w_j - learning_rate * (dMSE/dw_j + 2 * lambda * w_j)
```

### Lasso Regression

Lasso is commonly optimized using **coordinate descent**.

Coordinate descent updates one weight at a time while keeping other weights fixed.

### Lasso Coordinate Descent Intuition

For each weight:

1. Check how useful the feature is.
2. Apply L1 penalty.
3. Shrink the weight.
4. If the feature is weak, set the weight to zero.

This is why Lasso can perform feature selection.

### Lasso Soft Thresholding

The Lasso update uses a soft-thresholding idea:

```text
if z_j > lambda:
    w_j = z_j - lambda
elif z_j < -lambda:
    w_j = z_j + lambda
else:
    w_j = 0
```

This creates exact zero weights.

---

## 8. Training Workflow

```text
Input Data
down arrow
Clean Data
down arrow
Encode Categorical Features
down arrow
Scale Numerical Features
down arrow
Split Train and Test Data
down arrow
Initialize Ridge/Lasso Model
down arrow
Predict
down arrow
Compute Loss
down arrow
Apply Regularization Penalty
down arrow
Optimize Weights
down arrow
Repeat Until Converged
down arrow
Evaluate Model
down arrow
Save Best Model
```

For this project:

1. Load `CarPrice_Assignment.csv`.
2. Drop `car_ID` and `CarName`.
3. Separate features and target `price`.
4. One-hot encode categorical columns.
5. Scale numerical columns.
6. Train Ridge and Lasso.
7. Tune `alpha`.
8. Compare R2, MAE, MSE, and RMSE.
9. Save the better model as `ridge_lasso_model.pkl`.

---

## 9. Worked Numerical Example

Suppose we predict car price using one feature:

```text
x = engine size
y = car price
```

Small example dataset:

| Row | Engine Size `x` | Price `y` |
|---|---:|---:|
| 1 | 1 | 10 |
| 2 | 2 | 20 |
| 3 | 3 | 30 |

Assume initial model:

```text
y_hat = 0 + 5x
```

So:

```text
b = 0
w = 5
```

### Prediction for `x = 2`

```text
y_hat = 0 + 5*2
y_hat = 10
```

Actual value:

```text
y = 20
```

Error:

```text
error = y - y_hat
error = 20 - 10
error = 10
```

Squared error:

```text
error^2 = 10^2
error^2 = 100
```

### Ridge Penalty

Let:

```text
lambda = 0.1
w = 5
```

Ridge penalty:

```text
lambda * w^2 = 0.1 * 5^2
lambda * w^2 = 0.1 * 25
lambda * w^2 = 2.5
```

Total Ridge loss for this one row:

```text
100 + 2.5 = 102.5
```

### Lasso Penalty

Lasso penalty:

```text
lambda * |w| = 0.1 * |5|
lambda * |w| = 0.5
```

Total Lasso loss for this one row:

```text
100 + 0.5 = 100.5
```

### What This Shows

Both models care about prediction error.

Ridge adds a squared weight penalty.

Lasso adds an absolute weight penalty.

Ridge strongly punishes very large weights because the weight is squared.

---

## 10. Full Manual Training Example

Use three rows:

| Row | `x` | `y` |
|---|---:|---:|
| 1 | 1 | 10 |
| 2 | 2 | 20 |
| 3 | 3 | 30 |

Use a simple one-feature model:

```text
y_hat = b + wx
```

Initial values:

```text
w = 0
b = 0
learning_rate = 0.01
lambda = 0.1
```

### Iteration 1 Predictions

| Row | `x` | `y` | `y_hat = 0 + 0*x` |
|---|---:|---:|---:|
| 1 | 1 | 10 | 0 |
| 2 | 2 | 20 | 0 |
| 3 | 3 | 30 | 0 |

### Errors

Use:

```text
error = y - y_hat
```

| Row | `y` | `y_hat` | Error |
|---|---:|---:|---:|
| 1 | 10 | 0 | 10 |
| 2 | 20 | 0 | 20 |
| 3 | 30 | 0 | 30 |

### MSE Loss

```text
MSE = (1/3) * (10^2 + 20^2 + 30^2)
MSE = (1/3) * (100 + 400 + 900)
MSE = 1400/3
MSE = 466.67
```

### Ridge Loss

Because `w = 0`:

```text
Ridge penalty = lambda * w^2
Ridge penalty = 0.1 * 0^2
Ridge penalty = 0
```

```text
Ridge loss = 466.67 + 0
Ridge loss = 466.67
```

### Gradient for Weight

For MSE:

```text
dMSE/dw = (-2/n) * sum(x_i * error_i)
```

Calculate:

```text
sum(x_i * error_i) = (1*10) + (2*20) + (3*30)
sum(x_i * error_i) = 10 + 40 + 90
sum(x_i * error_i) = 140
```

```text
dMSE/dw = (-2/3) * 140
dMSE/dw = -93.33
```

Ridge gradient:

```text
dJ/dw = dMSE/dw + 2 * lambda * w
dJ/dw = -93.33 + 2 * 0.1 * 0
dJ/dw = -93.33
```

### Gradient for Bias

```text
dMSE/db = (-2/n) * sum(error_i)
```

```text
sum(error_i) = 10 + 20 + 30 = 60
```

```text
dMSE/db = (-2/3) * 60
dMSE/db = -40
```

### Update Weights

```text
w_new = w - learning_rate * dJ/dw
w_new = 0 - 0.01 * (-93.33)
w_new = 0.9333
```

```text
b_new = b - learning_rate * dMSE/db
b_new = 0 - 0.01 * (-40)
b_new = 0.4
```

After iteration 1:

```text
w = 0.9333
b = 0.4
```

### Iteration 2 Predictions

Use:

```text
y_hat = 0.4 + 0.9333x
```

| Row | `x` | `y` | `y_hat` |
|---|---:|---:|---:|
| 1 | 1 | 10 | 1.3333 |
| 2 | 2 | 20 | 2.2666 |
| 3 | 3 | 30 | 3.1999 |

### Iteration 2 Errors

| Row | `y` | `y_hat` | Error |
|---|---:|---:|---:|
| 1 | 10 | 1.3333 | 8.6667 |
| 2 | 20 | 2.2666 | 17.7334 |
| 3 | 30 | 3.1999 | 26.8001 |

### Iteration 2 MSE

```text
MSE = (1/3) * (8.6667^2 + 17.7334^2 + 26.8001^2)
MSE = (1/3) * (75.11 + 314.47 + 718.25)
MSE = 369.28
```

The loss decreased from `466.67` to about `369.28`, so the model learned in the correct direction.

---

## 11. Parameters - Learned Values

| Parameter | Meaning |
|---|---|
| Weight / coefficient | Importance of a feature |
| Bias / intercept | Base prediction when feature contribution is zero |
| Ridge coefficient | Weight after L2 shrinkage |
| Lasso coefficient | Weight after L1 shrinkage, possibly zero |

### Interpretation

If a coefficient is positive:

- increasing that feature increases predicted price

If a coefficient is negative:

- increasing that feature decreases predicted price

If a Lasso coefficient is zero:

- Lasso removed that feature from the model

---

## 12. Hyperparameters

| Hyperparameter | Used In | Purpose |
|---|---|---|
| `alpha` | Ridge and Lasso | Controls regularization strength |
| `max_iter` | Lasso | Maximum optimization iterations |
| `tol` | Ridge and Lasso | Stopping tolerance |
| `solver` | Ridge | Optimization algorithm |
| `fit_intercept` | Ridge and Lasso | Whether to learn bias term |
| `random_state` | Lasso in sklearn | Makes results reproducible |

---

## 13. Why Each Hyperparameter Exists

### Alpha

Problem:

- Without regularization, weights can become too large.

Solution:

- `alpha` controls how strongly weights are penalized.

Small `alpha`:

- weak regularization
- model behaves closer to Linear Regression
- may overfit

Large `alpha`:

- strong regularization
- smaller weights
- may underfit

### Max Iterations

Problem:

- Lasso may need many optimization steps to converge.

Solution:

- `max_iter` sets the maximum number of coordinate descent updates.

### Tolerance

Problem:

- Training should stop when improvement becomes very small.

Solution:

- `tol` controls when the optimizer stops.

### Solver

Problem:

- Different datasets need different numerical methods.

Solution:

- Ridge provides solvers for dense data, sparse data, small data, and large data.

---

## 14. Assumptions

Ridge and Lasso inherit many assumptions from Linear Regression.

### Main Assumptions

- Relationship between features and target is approximately linear.
- Rows are independent.
- Features are measured correctly.
- Important features are included.
- Errors are roughly centered around zero.
- Extreme outliers are handled.

### Ridge-Specific Notes

Ridge works well when many features are useful and correlated.

Example:

- engine size and horsepower both affect car price
- Ridge can keep both and shrink their weights

### Lasso-Specific Notes

Lasso works well when only some features are truly important.

Example:

- if some car specifications add little value, Lasso can set their coefficients to zero

---

## 15. Data Requirements

| Requirement | Needed? | Why |
|---|---|---|
| Numerical features | Yes | Regression needs numeric input after preprocessing |
| Categorical encoding | Yes | Text categories must be converted to numbers |
| Feature scaling | Yes | Regularization depends on coefficient size |
| Missing value handling | Yes | sklearn models do not accept missing values by default |
| Outlier checking | Recommended | Outliers can distort coefficients |
| Train/test split | Yes | Needed to measure generalization |

### Why Scaling Is Very Important

Ridge and Lasso penalize weights.

If one feature has a huge scale, such as curb weight, and another has a small scale, such as bore ratio, the penalty can become unfair.

Scaling makes features comparable.

---

## 16. Complexity

Let:

- `n` = number of rows
- `p` = number of features after encoding
- `k` = number of iterations

### Training Complexity

Ridge closed-form solution:

```text
O(p^3 + np^2)
```

Gradient-based Ridge:

```text
O(k * n * p)
```

Lasso coordinate descent:

```text
O(k * n * p)
```

### Prediction Complexity

For one row:

```text
O(p)
```

For `m` rows:

```text
O(m * p)
```

### Memory Complexity

```text
O(n * p)
```

The model coefficients themselves require:

```text
O(p)
```

---

## 17. Decision Boundary

Ridge and Lasso are regression models, so they do not have a classification decision boundary.

Instead, they learn a **prediction line**, **plane**, or **hyperplane**.

### For One Feature

```text
price
  ^
  |
30|                         *
25|                    *
20|               *
15|          *
10|     *
  +------------------------------> engine size
        prediction line
```

### Is It Linear?

Yes. Ridge and Lasso are linear models in terms of learned coefficients.

They can model nonlinear patterns only if nonlinear features are manually added, such as:

- polynomial features
- interaction terms
- log transforms

---

## 18. Overfitting and Underfitting

### Overfitting

Overfitting means the model performs well on training data but poorly on test data.

Signs:

- high training R2
- low test R2
- very large coefficients
- predictions change too much for small input changes

Ridge and Lasso reduce overfitting by shrinking weights.

### Underfitting

Underfitting means the model is too simple.

Signs:

- low training R2
- low test R2
- errors are large everywhere

Very large `alpha` can cause underfitting because the model shrinks weights too much.

---

## 19. Regularization

Regularization adds a penalty to the loss function to control model complexity.

### L1 Regularization - Lasso

```text
lambda * sum(|w_j|)
```

Effects:

- can make weights exactly zero
- performs feature selection
- useful when many features are irrelevant
- may keep only one feature from a group of correlated features

### L2 Regularization - Ridge

```text
lambda * sum(w_j^2)
```

Effects:

- shrinks weights smoothly
- usually does not make weights exactly zero
- handles multicollinearity well
- keeps correlated features together

### Elastic Net

Elastic Net combines L1 and L2:

```text
Elastic Net Loss = MSE + lambda1 * sum(|w_j|) + lambda2 * sum(w_j^2)
```

It is useful when:

- there are many features
- features are correlated
- some features should be removed
- Ridge alone keeps too many features
- Lasso alone is too aggressive

---

## 20. Feature Importance

Ridge and Lasso can be interpreted using coefficients.

### Positive Coefficient

Feature increases predicted price.

Example:

```text
horsepower coefficient > 0
```

Higher horsepower may increase predicted price.

### Negative Coefficient

Feature decreases predicted price.

Example:

```text
highwaympg coefficient < 0
```

Higher mileage may be associated with lower price in some datasets, depending on relationships.

### Zero Coefficient

In Lasso:

```text
coefficient = 0
```

means the feature was removed.

### Important Warning

Feature importance is meaningful only after preprocessing is understood.

For one-hot encoded categorical features, each category gets its own coefficient.

---

## 21. Advantages

### Ridge Regression

- Reduces overfitting.
- Handles multicollinearity well.
- Keeps all features.
- More stable than plain Linear Regression.
- Good when many features are useful.

### Lasso Regression

- Reduces overfitting.
- Can remove irrelevant features.
- Produces sparse models.
- Helps with feature selection.
- Easier to interpret when many weights become zero.

### Both

- Fast to train.
- Simple to understand.
- Work well as strong baseline models.
- More interpretable than many complex models.

---

## 22. Limitations

### Ridge Limitations

- Does not remove features completely.
- Still assumes linear relationships.
- Coefficients can be hard to interpret after one-hot encoding.

### Lasso Limitations

- Can be unstable with highly correlated features.
- May randomly select one feature from a correlated group.
- Needs careful scaling.
- May underfit if `alpha` is too large.

### Shared Limitations

- Sensitive to outliers.
- Cannot automatically learn complex nonlinear patterns.
- Requires preprocessing for categorical data.
- Performance depends on feature quality.

---

## 23. Failure Cases

Ridge and Lasso can fail when:

- the true relationship is highly nonlinear
- important features are missing
- many outliers exist
- categorical variables are badly encoded
- train and test data come from different distributions
- regularization strength is poorly chosen
- features are not scaled

### Example Failure

If car price depends on complex interactions such as:

```text
brand * engine type * market segment
```

a basic Ridge or Lasso model may miss that unless interaction features are created.

---

## 24. Edge Cases

### Dataset Is Tiny

Ridge is often safer because it stabilizes weights.

Lasso may remove too many features.

### Dataset Is Huge

Both can work well, but solver choice matters.

For very large datasets, iterative solvers are preferred.

### Features Are More Than Samples

Ridge handles this better than Linear Regression.

Lasso can also help by selecting features.

### Imbalanced Classes

This is mainly a classification issue.

For regression, the similar problem is an imbalanced target distribution.

Example:

- many cheap cars
- very few expensive cars

The model may predict cheap and mid-range cars better than luxury cars.

### Missing Values

Ridge and Lasso do not handle missing values directly in standard sklearn usage.

Missing values should be handled using:

- mean/median imputation for numeric features
- mode/constant imputation for categorical features

### Unseen Categories

The project uses:

```text
OneHotEncoder(handle_unknown="ignore")
```

This prevents crashes when the app receives a category not seen during training.

---

## 25. Evaluation Metrics

This is a regression project, so the important metrics are regression metrics.

### MAE - Mean Absolute Error

```text
MAE = (1/n) * sum(|y_i - y_hat_i|)
```

Meaning:

- average absolute prediction error
- easy to understand

Example:

```text
MAE = 2000
```

means predictions are off by about 2000 currency units on average.

### MSE - Mean Squared Error

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

Meaning:

- squares errors
- punishes large errors heavily

### RMSE - Root Mean Squared Error

```text
RMSE = sqrt(MSE)
```

Meaning:

- similar unit as the target
- punishes large errors more than MAE

### R2 Score

```text
R2 = 1 - (SS_res / SS_tot)
```

Meaning:

- measures how much variance the model explains

Interpretation:

- `R2 = 1` means perfect prediction
- `R2 = 0` means model is no better than predicting the mean
- `R2 < 0` means model is worse than predicting the mean

### Classification Metrics

Accuracy, precision, recall, and F1 are not used here because Ridge and Lasso Regression predict continuous values.

---

## 26. Comparison with Similar Models

| Feature | Linear Regression | Ridge Regression | Lasso Regression |
|---|---|---|---|
| Regularization | No | Yes, L2 | Yes, L1 |
| Penalty | None | `sum(w_j^2)` | `sum(|w_j|)` |
| Shrinks weights | No | Yes | Yes |
| Makes weights zero | No | Usually no | Yes |
| Feature selection | No | No | Yes |
| Handles multicollinearity | Poorly | Well | Sometimes unstable |
| Interpretability | High | High | High when sparse |
| Overfitting control | Weak | Strong | Strong |
| Best use case | Simple baseline | Many useful correlated features | Many irrelevant features |

### Ridge vs Lasso

Use Ridge when:

- most features are useful
- features are correlated
- you want stable coefficients

Use Lasso when:

- only some features are useful
- you want automatic feature selection
- you want a sparse model

Use Elastic Net when:

- you want both Ridge stability and Lasso feature selection

---

## 27. Real-World Applications

### Finance

- house price prediction
- car price prediction
- credit risk score estimation
- sales forecasting

### Healthcare

- predicting medical costs
- estimating patient recovery time
- predicting blood pressure or lab values

### Marketing

- customer lifetime value prediction
- advertisement spend analysis
- demand forecasting

### Manufacturing

- predicting product quality scores
- estimating machine maintenance cost
- forecasting production output

### Education

- predicting exam scores
- estimating student performance index

---

## 28. Scikit-Learn Implementation

### Ridge Regression

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Lasso Regression

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0, max_iter=100000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### With Preprocessing Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
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
        ("model", Ridge(alpha=1.0)),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="r2",
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

---

## 29. Interview Questions

### Basic Questions

1. What is Ridge Regression?
2. What is Lasso Regression?
3. How are Ridge and Lasso different from Linear Regression?
4. What problem does regularization solve?
5. What is the difference between L1 and L2 regularization?

### Mathematical Questions

1. Write the Ridge Regression loss function.
2. Write the Lasso Regression loss function.
3. Why does Ridge shrink coefficients?
4. Why can Lasso make coefficients exactly zero?
5. What is the role of `alpha`?

### Practical Questions

1. Why is feature scaling important for Ridge and Lasso?
2. When would you choose Ridge over Lasso?
3. When would you choose Lasso over Ridge?
4. What happens if `alpha` is too small?
5. What happens if `alpha` is too large?

### Model Behavior Questions

1. How does Ridge handle multicollinearity?
2. Why can Lasso be unstable with correlated features?
3. Can Ridge perform feature selection?
4. Can Lasso underfit?
5. How do you interpret coefficients after one-hot encoding?

### Evaluation Questions

1. Which metrics are used for regression?
2. What does RMSE tell us?
3. What does R2 score mean?
4. Why might MAE be easier to explain than MSE?
5. How do you detect overfitting?

### Project Questions

1. What is the target variable in this project?
2. Why did we drop `car_ID`?
3. Why did we drop `CarName`?
4. Why did we use one-hot encoding?
5. Why did we compare both Ridge and Lasso?

---

## 30. Summary

Ridge and Lasso are regularized versions of Linear Regression.

Ridge adds an L2 penalty:

```text
lambda * sum(w_j^2)
```

Lasso adds an L1 penalty:

```text
lambda * sum(|w_j|)
```

Ridge is best when many features are useful and correlated.

Lasso is best when only some features are important and feature selection is useful.

Both models are simple, fast, interpretable, and powerful baselines for regression tasks.

In this project, Ridge and Lasso are used to predict car prices from vehicle specifications, and the best model is saved for use in the Streamlit app.
