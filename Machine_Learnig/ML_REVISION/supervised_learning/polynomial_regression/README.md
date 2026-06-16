# Polynomial Regression - House Price Prediction

## 1. Model Name

- **Full Name:** Polynomial Regression (Degree 2)
- **Category:** Regression, Nonlinear Extension of Linear Regression
- **Learning Type:** Supervised Learning, Parametric
- **Output Type:** Continuous numerical values

---

## 2. Problem It Solves

- **Task:** Predicting continuous values using polynomial feature transformations
- **Use Case:** Predicting house prices based on square footage
- **Why We Need It:**
  - Linear relationships insufficient for many real-world problems
  - Price doesn't scale linearly with size (diminishing returns)
  - Captures curved relationships automatically
  - Bridges gap between linear simplicity and complex nonlinear models
  - Interpretable alternative to black-box nonlinear models

---

## 3. Intuition

### Simple Explanation
Polynomial Regression extends Linear Regression by fitting a curved line instead of a straight line. It transforms features into higher powers (x², x³, etc.) and fits a linear model on these transformed features.

### Real-World Analogy
**Linear Regression:** "Price increases by $5K per 100 sqft everywhere"
- Straight line, same rate everywhere
- House 1000 sqft: $50K, House 4000 sqft: $200K

**Polynomial Regression:** "Price increases by $5K per 100 sqft for smaller houses, but less for larger houses"
- Curved line, rate changes
- Captures diminishing returns
- House 1000 sqft: $50K, House 4000 sqft: $180K (not $200K)

### Visual Comparison

```
Linear (y = β₀ + β₁x):
     |     /
     |    /  Straight line
     |   /
     |__/____________

Polynomial (y = β₀ + β₁x + β₂x²):
     |    __
     |   /  \  Curved line
     |  /    \
     |________\____
```

---

## 4. Mathematical Foundation

### Polynomial Model (Degree 2)

$$y = \beta_0 + \beta_1 x + \beta_2 x^2$$

where:
- $\beta_0$ = intercept
- $\beta_1$ = linear coefficient
- $\beta_2$ = quadratic coefficient
- $x$ = square footage
- $y$ = price

### General Polynomial (Degree n)

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_n x^n$$

### Feature Transformation

Original features: $[x]$

Transformed (degree 2): $[1, x, x^2]$

Transformed (degree 3): $[1, x, x^2, x^3]$

Then fit standard linear regression on transformed features.

### Loss Function

Minimize Mean Squared Error on transformed features:

$$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

$$= \frac{1}{2m} \sum_{i=1}^{m} (\beta_0 + \beta_1 x_i + \beta_2 x_i^2 - y_i)^2$$

### Solution via Normal Equation

$$\beta = (X^T X)^{-1} X^T y$$

where $X$ includes polynomial features.

---

## 5. Detailed Example: House Prices

### Training Data

| House | Sqft (x) | Price (y) |
|-------|----------|-----------|
| A | 1000 | $100K |
| B | 2000 | $200K |
| C | 3000 | $245K |
| D | 4000 | $240K |

### Observations
- 1000→2000 sqft: +$100K increase
- 2000→3000 sqft: +$45K increase (slowdown!)
- 3000→4000 sqft: -$5K decrease (diminishing!)

This is **nonlinear** → Polynomial needed

### Linear Fit (Wrong)

Fit line through data:
$$y = 80 + 0.05x$$

Predictions:
- 1000 sqft: $130K ✗ (actual $100K, error $30K)
- 2000 sqft: $180K ✗ (actual $200K, error -$20K)
- 3000 sqft: $230K ✗ (actual $245K, error -$15K)
- 4000 sqft: $280K ✗ (actual $240K, error $40K)

**RMSE: $31.2K** (poor fit)

### Polynomial Fit (Better)

Fit curve with quadratic term:
$$y = \beta_0 + \beta_1 x + \beta_2 x^2$$

Using normal equation on transformed features $[1, x, x^2]$:

Solved: $\beta_0 = 50$, $\beta_1 = 0.15$, $\beta_2 = -0.000050$

Model:
$$y = 50 + 0.15x - 0.000050x^2$$

Predictions:
- 1000 sqft: $50 + 150 - 50 = $150K ✗ (still off)
- 2000 sqft: $50 + 300 - 200 = $150K ✗
- 3000 sqft: $50 + 450 - 450 = $50K ✗
- 4000 sqft: $50 + 600 - 800 = -$150K ✗

**RMSE: $165K** (worse! Need better fitting)

### Proper Polynomial Fit (Using Least Squares)

With proper optimization:
$$y = 5 + 0.12x - 0.00002x^2$$

Predictions:
- 1000 sqft: $5 + 120 - 20 = $105K ✓ (close!)
- 2000 sqft: $5 + 240 - 80 = $165K ✗
- 3000 sqft: $5 + 360 - 180 = $185K ✗
- 4000 sqft: $5 + 480 - 320 = $165K ✗

Better but still not perfect (real data has noise).

---

## 6. Why Polynomial Features Work

### The Key Insight

Linear Regression finds the best **line** through data.
Polynomial Regression finds the best **curve** through data.

By creating $x^2$ feature:
- Model can learn nonlinear relationships
- Each coefficient adjusted for both linear and quadratic effects
- Still interpretable as linear model (on transformed features)

### Example: Real House Data

```
Actual relationship: Price increases with size, but slower for large houses
- At 1000 sqft: +$100K per 1000 sqft increase
- At 3000 sqft: +$30K per 1000 sqft increase
- At 5000 sqft: +$5K per 1000 sqft increase

This is captured by β₂x² term:
- Positive β₁ pushes price up
- Negative β₂x² slows growth for large x
- Combined: curve shape
```

---

## 7. Degree Selection

### Degree 1 (Linear)

$$y = \beta_0 + \beta_1 x$$

- **Pros:** Simple, fast, few parameters, low overfitting risk
- **Cons:** Can't capture curvature
- **When:** Data clearly linear

### Degree 2 (Quadratic)

$$y = \beta_0 + \beta_1 x + \beta_2 x^2$$

- **Pros:** Captures common bowl/inverted-bowl shapes, balanced
- **Cons:** Assumes specific shape
- **When:** Single peak or valley in relationship ← **Most common**

### Degree 3 (Cubic)

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$$

- **Pros:** More flexibility
- **Cons:** More parameters, overfitting risk
- **When:** Data has S-shape or multiple turning points

### Degree 4+

- **Pros:** Very flexible
- **Cons:** Rapidly overfits, oscillates wildly
- **When:** Rarely needed for real data

### Visual: Different Degrees

```
Degree 1: ___/____    Degree 2:  __/\__    Degree 3: __/\____
Linear               Quadratic            Cubic (S-curve)

Degree 4: Wiggly    Degree 5: Very
squiggly line        squiggly oscillations

⚠️ High degrees → Overfitting!
```

---

## 8. Advantages

✅ **Captures Nonlinear Relationships**
- Price diminishing returns naturally captured
- Curved boundaries in feature space

✅ **Interpretable**
- Coefficients explain effect of each polynomial term
- Still linear regression (on transformed features)
- Easier than black-box methods

✅ **Flexible**
- By adjusting degree, balance simplicity vs fit
- Can approximate any smooth function (Weierstrass approximation)

✅ **Fast Training**
- Linear regression on transformed features
- Closed-form solution exists (normal equation)
- No iterative optimization needed

✅ **No Scaling Required**
- Can work without feature scaling
- Better numerical stability with scaling though

✅ **Low Data Requirement**
- Fewer samples needed than deep learning
- Degree 2-3 usually enough for most problems

---

## 9. Limitations

❌ **Prone to Overfitting**
- High degree polynomials fit training noise
- Poor generalization beyond training range
- Oscillates wildly outside training region

❌ **Limited Extrapolation**
- Can't predict well outside training range
- High-degree polynomials explode at extremes
- Example: Degree 4 fit extrapolates to ±∞

❌ **Curse of Dimensionality**
- With multiple features: (sqft, bedrooms, bathrooms)
- Degree 2 creates 10+ features from 3 inputs
- Degree 3 creates 20+ features
- Exponential growth with features and degree

❌ **Not Universal**
- Assumes smooth curved relationship
- Can't handle step functions or discontinuities
- May be inappropriate for categorical-like relationships

❌ **Numerical Instability**
- Large x values create very large x² values
- Scaling necessary to avoid computational issues
- Matrix inversion can be numerically unstable

---

## 10. Overfitting & Underfitting

### Underfitting (Degree Too Low)

**Symptoms:**
- High training error
- High test error
- Same poor performance on both

**Example:** Degree 1 on curved data
```
Training RMSE: $50K
Test RMSE: $52K
```

**Solution:** Increase degree

### Good Fit (Right Degree)

**Symptoms:**
- Moderate training error
- Similar test error
- Good generalization

**Example:** Degree 2 on curved data
```
Training RMSE: $15K
Test RMSE: $16K ← Close! Good fit
```

### Overfitting (Degree Too High)

**Symptoms:**
- Very low training error
- Much higher test error
- Large train/test gap

**Example:** Degree 5 on curved data
```
Training RMSE: $2K ← Perfect fit (memorized!)
Test RMSE: $60K ← Terrible generalization
```

**Solution:** Decrease degree, use regularization

### Visual: Overfitting

```
Data with noise (Blue dots):

Degree 2:          Degree 5:
  \  /\             /\/\
   \/  \___   ✓     /    \/\  ✗ Wiggly!
                    
Good smooth curve   Oscillates to fit every point
```

---

## 11. Regularization: Ridge & Lasso on Polynomials

### Problem with High Degree

```
Degree 10 polynomial:
β = [50, 200, -150, 80, -30, 10, -3, 1, -0.2, 0.05, -0.001]

Large coefficients → Unstable predictions
```

### Ridge Regression (L2 Penalty)

$$J(\beta) = \frac{1}{2m} \sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j \beta_j^2$$

- Shrinks large coefficients toward 0
- Keeps all features
- Smoother predictions

### Lasso Regression (L1 Penalty)

$$J(\beta) = \frac{1}{2m} \sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j |\beta_j|$$

- Shrinks some coefficients to exactly 0
- Automatic feature selection
- Sparse model

### When to Use

- Standard Polynomial: Small degree (2-3)
- Ridge Polynomial: Medium degree (4-5) with multicollinearity
- Lasso Polynomial: High degree (5+) to select features

---

## 12. Multicollinearity

### Problem: Correlated Features

Polynomial features highly correlated:
- $x$ ranges [1000, 5000]
- $x^2$ ranges [1M, 25M]
- $x$ and $x^2$ strongly correlated: $r = 0.995$

**Effect:**
- Coefficients unstable
- Small data changes → large coefficient changes
- Large standard errors
- Poor predictions

### Solution: Scaling

StandardScaler transforms to mean=0, std=1:
- Reduces magnitude differences
- Improves numerical stability
- Reduces multicollinearity effects

### Example

**Before scaling:**
- $x$ ranges [1000, 5000]
- $x^2$ ranges [1M, 25M]
- Coefficients $\beta_1 = 100$, $\beta_2 = -0.001$ (vastly different magnitudes)

**After scaling:**
- $x$ ranges [-1.5, 1.5]
- $x^2$ ranges [0, 2.25]
- Coefficients $\beta_1 = 20$, $\beta_2 = -15$ (similar magnitudes, more stable)

---

## 13. Data Requirements

| Requirement | Needed | Notes |
|---|---|---|
| **Sample Size** | $n > 5 \times (\text{degree} + 1)$ | More samples → higher degree feasible |
| **Feature Quality** | Continuous | Polynomial assumes smooth input |
| **Outliers** | Handle carefully | Affect curve fitting greatly |
| **Missing Values** | None | Must impute or remove |
| **Multicollinearity** | Scale features | Use StandardScaler |
| **Target Distribution** | Any | No specific requirement |

### For House Prices Example

- Degree 2 needs: $n > 5 \times 3 = 15$ samples
- Recommend: $n > 50$ samples for stability

---

## 14. Complexity Analysis

### Time Complexity

#### Training
$$O(n \times p^3)$$

where:
- $n$ = number of samples
- $p$ = number of polynomial features (including intercept)
- $p^3$ from matrix inversion in normal equation

For degree 2 with 1 feature:
- $p = 3$ (features: 1, x, x²)
- Complexity: $O(n)$ (fast!)

#### Prediction
$$O(p)$$

Simple dot product with coefficients.

### Space Complexity

$$O(n \times p)$$

Store transformed feature matrix.

---

## 15. Evaluation Metrics

### Regression Metrics

**Mean Absolute Error (MAE)**
$$MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|$$
- Interpretation: "Average prediction off by $20K"
- Robust to outliers

**Root Mean Squared Error (RMSE)**
$$RMSE = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$$
- Interpretation: "Typical error $25K"
- Penalizes large errors

**R² Score**
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- Interpretation: "Model explains 85% of price variation"
- 0 to 1 scale

### Example Evaluation

```
Degree 1 (Linear):
  RMSE: $35K, R²: 0.82

Degree 2 (Quadratic): ← Best
  RMSE: $15K, R²: 0.96

Degree 3 (Cubic):
  RMSE: $14K, R²: 0.97 (but overfitting?)

Degree 4:
  RMSE: $12K, R²: 0.98 (likely overfitting)
```

---

## 16. Scikit-Learn Implementation

### Basic Usage

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Fit model
model = LinearRegression()
model.fit(X_poly_scaled, y)

# Predict
predictions = model.predict(X_poly_scaled)
```

### With Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    LinearRegression(),
    X_poly_scaled,
    y,
    cv=5,
    scoring='r2'
)

print(f"CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Hyperparameter Tuning: Degree

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

degrees = [1, 2, 3, 4, 5]
best_degree = None
best_score = -np.inf

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    X_poly_scaled = scaler.fit_transform(X_poly)
    
    model = LinearRegression()
    scores = cross_val_score(model, X_poly_scaled, y_train, cv=5)
    
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_degree = degree

print(f"Best degree: {best_degree}")
```

---

## 17. Comparison with Linear Regression

| Aspect | Linear | Polynomial |
|--------|--------|-----------|
| **Relationship** | Straight line | Curved line |
| **Flexibility** | Low | Medium |
| **Parameters** | 2 (1 feature) | 3+ (1 feature) |
| **Overfitting Risk** | Low | High |
| **Interpretability** | Highest | High |
| **Speed** | Faster | Slightly slower |
| **Use When** | Clearly linear | Curved relationship |

---

## 18. Comparison with Nonlinear Models

| Aspect | Polynomial | Decision Tree | Neural Network |
|--------|-----------|---|---|
| **Interpretability** | High | Highest | Low |
| **Extrapolation** | Possible (risky) | No | Possible |
| **Training Speed** | Fast | Fast | Slow |
| **Flexibility** | Medium | High | Very High |
| **Data Needed** | Little | Moderate | Much |
| **Default Choice** | Good | Better | Only for big data |

---

## 19. Real-World Applications

### Finance
- **Stock Price Trends:** Polynomial fit to historical prices
- **Portfolio Returns:** Nonlinear risk-return relationship
- **Loan Interest:** Rate vs credit score relationship

### Physics & Engineering
- **Motion Equations:** s = ut + ½at² (naturally quadratic!)
- **Growth Models:** Exponential-like curves
- **Bridge Deflection:** Load vs deflection (polynomial)

### Economics
- **Supply Curves:** Price vs quantity nonlinear
- **Demand Elasticity:** Demand vs price with diminishing effect
- **Cost Functions:** Costs increase nonlinearly with production

### Real Estate ← Our Use Case
- **Property Valuation:** Price vs size with diminishing returns
- **Rent vs Location:** Distance from city center (squared)
- **Building Height:** Cost doesn't scale linearly

### Biology
- **Plant Growth:** Height vs time (sigmoidal, approximated by polynomial)
- **Drug Response:** Dose vs effect (often curved)
- **Population Growth:** Limited by resources (cubic shape)

---

## 20. Interview Questions

### Q1: When Would You Use Polynomial Over Linear?
**Answer:**
- When data shows curved relationship (scatterplot non-linear)
- When single feature effect changes with value (diminishing returns)
- When simple linear model has high error
- Trade-off: Simplicity (linear) vs fit (polynomial)
- Check: Try degree 1, 2, 3 via cross-validation, pick best

### Q2: Why Scale Polynomial Features?
**Answer:**
- Polynomial features have different magnitudes (x vs x²)
- x ranges [1000, 5000], x² ranges [1M, 25M]
- Large magnitude differences → numerical instability
- Scaling centers at 0, scales to unit variance
- Improves: computation stability, coefficient interpretation

### Q3: Explain Degree Selection
**Answer:**
- Degree 1: Linear (can underfitting on curved data)
- Degree 2: Quadratic, most common (single peak/valley)
- Degree 3+: Cubic+ (more flexible but overfitting risk)
- Process: Try degrees 1,2,3 via 5-fold CV, pick highest score
- Rule: Start with degree 2, increase only if needed

### Q4: What's Overfitting in Polynomial Context?
**Answer:**
- High degree polynomial fits training noise
- Example: Degree 5 gets training R²=0.99, test R²=0.60
- Oscillates wildly outside training range
- Solution: Low degree, cross-validation, regularization (Ridge/Lasso)

### Q5: Can't Polynomial Do Everything?
**Answer:**
- High degree → exponential overfitting risk
- Can't handle discontinuities or step functions
- Requires smooth relationship assumption
- Limited to univariate nonlinearity
- For complex: Decision Trees, Random Forest, Neural Nets

### Q6: How Decide between Polynomial and Others?
**Answer:**
1. Try Linear first (baseline)
2. If poor fit, try Polynomial degree 2-3
3. If still poor, try Decision Tree / Random Forest
4. Only use complex models if really needed
5. Occam's Razor: Simplest model that works well

### Q7: Extrapolation Risk?
**Answer:**
- Linear: Extrapolates reasonably (same slope)
- Polynomial: Explodes at extremes
- Example: Degree 4, fitted on [0, 100], predicts -∞ at x=-10
- Solution: Use lower degree, confine to training range
- Best: Ensemble or piecewise models

### Q8: Handle Multiple Features (sqft, bedrooms, baths)?
**Answer:**
- Degree 2 creates: sqft, sqft², bedrooms, bedrooms², baths, baths²
- Plus interactions: sqft×bedrooms, etc.
- Curse of dimensionality: Features explode
- Solutions:
  - Use lower degree (degree 2 only)
  - Select important features first
  - Use Lasso for automatic selection
  - Consider tree-based models instead

### Q9: What if Data Has Outliers?
**Answer:**
- Polynomial fitting very sensitive to outliers
- One extreme point forces curve to fit it
- Solutions:
  - Outlier detection & removal
  - Robust regression (Huber, RANSAC)
  - Lower degree (regularizes implicitly)
  - Tree-based models (robust to outliers)

### Q10: Parametric vs Nonparametric?
**Answer:**
- Polynomial = Parametric
- Fixed number of parameters determined by degree
- Polynomial degree 2: Always 3 parameters
- vs Random Forest (nonparametric): Parameters depend on data
- Parametric advantage: Interpretable, stable
- Nonparametric advantage: Flexible, automatic selection

---

## Conclusion

Polynomial Regression is a powerful yet simple bridge between Linear Regression and complex nonlinear models:
- **Captures curved relationships** automatically
- **Remains interpretable** (explainable coefficients)
- **Fast to train** (closed-form solution)
- **Risk of overfitting** with high degrees
- **Perfect for** smooth, polynomial-like relationships

For most real-world problems, degree 2-3 provides the best balance of flexibility and generalization.

---

## Additional Resources

### Recommended Reading
- "An Introduction to Statistical Learning" - Chapter on Regression
- "Applied Predictive Modeling" - Nonlinear Regression chapter

### Online Resources
- [Scikit-Learn Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Polynomial Regression Tutorial](https://www.datacamp.com/courses/polynomial-regression)

### Related Models
- Linear Regression (degree 1)
- Ridge Regression (with polynomial features)
- Lasso Regression (with polynomial features)
- Spline Regression (piecewise polynomials)
- Decision Trees (nonparametric alternative)

---

**Model Location:** `polynomial_regression/` folder  
**Dataset:** House Prices Data  
**Target Variable:** Price (continuous, $K)  
**Feature:** Square Footage  
**Polynomial Degree:** 2 (Quadratic)  
**Application:** Real estate price prediction  
**Last Updated:** June 2026
