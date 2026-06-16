# Decision Tree Regressor — Complete Study Guide

## 1. Model Name

- **Full Name:** Decision Tree Regressor
- **Category:** Regression
- **Type:** Supervised Learning

---

## 2. Problem It Solves

### What task is it designed for?
Decision Tree Regressor predicts **continuous numerical values** by learning a set of if-else rules that partition the input space. Unlike the Classifier which predicts categories, the Regressor predicts a number — the average target value of the training samples in each leaf.

Examples:
- Predicting house prices based on size, location, and age
- Forecasting energy consumption based on temperature and time
- Estimating patient recovery time based on age and treatment type

### Why do we need this model?
- Many real-world prediction tasks have continuous outputs — a Classifier cannot handle these.
- Decision Tree Regressor can capture **non-linear relationships** without any transformation of features.
- It is highly interpretable: you can trace the exact rules that led to a prediction.
- It serves as the base learner for powerful ensemble methods like Gradient Boosting and Random Forest Regressor.

---

## 3. Intuition

### Project Context - Insurance Charges Prediction

In this folder, the Decision Tree Regressor is used for an **insurance cost regression** task.

The local dataset is:

```text
insurance.csv
```

The target column is:

```text
charges
```

The model predicts a continuous insurance charge amount from:

| Feature | Meaning |
|---|---|
| `age` | Customer age |
| `sex` | Customer sex |
| `bmi` | Body mass index |
| `children` | Number of children/dependents |
| `smoker` | Whether the customer smokes |
| `region` | Customer region |

For this project, a decision tree can learn pricing rules such as:

```text
if smoker is yes and bmi is high, predict higher charges
if smoker is no and age is low, predict lower charges
```

The saved model file is:

```text
decision_tree_insurance_model.pkl
```

### Simple Explanation
Imagine you are trying to estimate the price of a house. You start by asking: "Is the area greater than 1500 sq ft?" If yes → prices tend to be higher. Then ask: "Is it in a premium neighborhood?" If yes → price estimate goes up further. The final prediction is the **average price of all training houses** that fall into the same category (leaf node) as the new house.

A Decision Tree Regressor creates these nested rules automatically, segmenting the data space into rectangular regions and assigning each region a constant predicted value.

### Real-World Analogy
Think of a **real estate appraiser** who estimates home values by comparing the new property to known sales:
1. First, they narrow down by location (city/suburb)
2. Then by size (large/medium/small)
3. Then by condition (new/old)
4. Finally, they average the prices of homes in that same bucket → that's the estimate

The Decision Tree Regressor does exactly this — it creates buckets based on feature conditions and predicts the **mean of the training samples** in that bucket.

---

## 4. Mathematical Foundation

### Core Formula — Prediction

For a leaf node $L$, the predicted value for any sample that falls into $L$ is:

$$\hat{y} = \frac{1}{|L|} \sum_{i \in L} y_i$$

This is simply the **mean of the target values** of all training samples in that leaf.

### Core Formula — Split Criterion (Mean Squared Error)

$$MSE(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$

Where $\bar{y}_S$ is the mean of all targets in node $S$.

### Explain Every Term

| Symbol | Meaning |
|---|---|
| $\hat{y}$ | Predicted value for a new input sample |
| $L$ | A leaf node (terminal node) |
| $\|L\|$ | Number of training samples in leaf $L$ |
| $y_i$ | Actual target value of training sample $i$ |
| $S$ | Current node (set of samples at that node) |
| $\bar{y}_S$ | Mean of target values in node $S$ |
| $MSE(S)$ | Mean Squared Error (impurity measure for regression) |

---

## 5. Objective Function

The Decision Tree Regressor minimizes the **weighted Mean Squared Error** across child nodes at every split:

$$Cost(split) = \frac{|Left|}{|S|} \cdot MSE(Left) + \frac{|Right|}{|S|} \cdot MSE(Right)$$

**Why MSE?**
- MSE measures the **variance** of the target values within a node.
- A pure node (all samples with similar $y$ values) has low MSE.
- The goal: find the split that reduces variance the most.
- We want leaf nodes where all samples have similar target values → then the mean is a good prediction.

**When MSE is high:** The node contains samples with very different target values → the mean would be a poor predictor → we need to split further.

**When MSE is 0:** All samples in the node have the exact same target value → perfect leaf.

**Variance Reduction** (equivalent objective):

$$Variance\ Reduction = Var(S) - \left(\frac{|Left|}{|S|} \cdot Var(Left) + \frac{|Right|}{|S|} \cdot Var(Right)\right)$$

---

## 6. Loss Function Derivation

### Deriving the MSE Split Criterion

At each node, we want to find the split that makes the target values as homogeneous as possible within each child.

**Variance of a set $S$:**
$$Var(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

This is equivalent to MSE when predicting $\bar{y}$ for every sample in $S$.

**Why the mean minimizes MSE:**  
The prediction that minimizes $\sum (y_i - \hat{y})^2$ is the mean $\hat{y} = \bar{y}$.  
Proof:
$$\frac{\partial}{\partial \hat{y}} \sum_{i \in L} (y_i - \hat{y})^2 = -2 \sum_{i \in L} (y_i - \hat{y}) = 0$$
$$\Rightarrow \hat{y} = \frac{1}{|L|} \sum_{i \in L} y_i = \bar{y}$$

So using the mean as the leaf prediction is theoretically optimal for MSE loss.

### Alternative Criterion: Mean Absolute Error

$$MAE(S) = \frac{1}{|S|} \sum_{i \in S} |y_i - \tilde{y}_S|$$

Where $\tilde{y}_S$ is the **median** (which minimizes MAE). More robust to outliers but slower to compute.

### No Gradient Used
Decision Trees are greedy, rule-based algorithms. There is no gradient descent. The "optimization" is an exhaustive search over all features and thresholds.

---

## 7. Optimization Method

### Algorithm: CART for Regression

At each node $S$:
1. For every feature $f$ in the dataset:
   a. Sort the samples by feature $f$
   b. For every unique midpoint threshold $t$ between consecutive values:
      - Split: Left = $\{i : x_{i,f} \leq t\}$, Right = $\{i : x_{i,f} > t\}$
      - Compute: $Cost = \frac{|Left|}{|S|} \cdot MSE(Left) + \frac{|Right|}{|S|} \cdot MSE(Right)$
2. Choose the $(f^*, t^*)$ pair that minimizes $Cost$
3. Split the node on $(f^*, t^*)$
4. Recurse on Left and Right children

### Prediction Rule at Each Leaf:
$$\hat{y} = \bar{y}_{leaf} = \text{mean of } y_i \text{ for all training samples in that leaf}$$

### Stopping Criteria:
- `max_depth` is reached
- Node has fewer than `min_samples_split` samples
- Further splitting reduces MSE by less than `min_impurity_decrease`
- Node is already pure (all $y_i$ equal)

---

## 8. Training Workflow

```
Input: Labeled Training Data (features X, target y — continuous)
        ↓
Start at Root Node (all n samples)
        ↓
At each node:
  ┌─────────────────────────────────────────────┐
  │ For each feature f:                         │
  │   For each threshold t:                     │
  │     Split → Left, Right                     │
  │     Compute weighted MSE of split           │
  │   End for                                   │
  │ End for                                     │
  │ Choose (f*, t*) with minimum weighted MSE   │
  └─────────────────────────────────────────────┘
        ↓
    Split node into Left child and Right child
        ↓
    Recurse on each child
        ↓
    Until stopping criterion met
        ↓
Leaf Node → assign prediction = mean(y) of samples in leaf
        ↓
Tree is trained → ready to predict
```

### Prediction Workflow:
```
New Sample x
    ↓
Start at Root
    ↓
At each node: check x[feature] vs threshold
    ↓ Yes (≤ threshold)     ↓ No (> threshold)
  Left child             Right child
    ↓
Until leaf node reached
    ↓
Return: ŷ = mean(y) of leaf
```

---

## 9. Worked Numerical Example

### Dataset

| Sample | X: Size (sq ft) | y: Price ($k) |
|---|---|---|
| 1 | 1000 | 150 |
| 2 | 1500 | 200 |
| 3 | 2000 | 300 |
| 4 | 2500 | 400 |
| 5 | 3000 | 500 |

**Goal:** Find the best split on X (Size)

### Try Threshold: X ≤ 1750 (splits into 2 groups)

**Left:** Samples 1, 2 → y = [150, 200]  
$$\bar{y}_{Left} = 175, \quad MSE_{Left} = \frac{(150-175)^2 + (200-175)^2}{2} = \frac{625+625}{2} = 625$$

**Right:** Samples 3, 4, 5 → y = [300, 400, 500]  
$$\bar{y}_{Right} = 400, \quad MSE_{Right} = \frac{(300-400)^2 + (400-400)^2 + (500-400)^2}{3} = \frac{10000+0+10000}{3} = 6666.67$$

**Weighted MSE of Split:**
$$Cost = \frac{2}{5}(625) + \frac{3}{5}(6666.67) = 250 + 4000 = 4250$$

### Try Threshold: X ≤ 1250

**Left:** Sample 1 → y = [150] → $MSE_{Left} = 0$  
**Right:** Samples 2, 3, 4, 5 → y = [200, 300, 400, 500]  
$$\bar{y}_{Right} = 350, \quad MSE_{Right} = \frac{(200-350)^2+(300-350)^2+(400-350)^2+(500-350)^2}{4}$$
$$= \frac{22500+2500+2500+22500}{4} = 12500$$

**Weighted MSE:**
$$Cost = \frac{1}{5}(0) + \frac{4}{5}(12500) = 10000$$

**Threshold 1750 gives lower cost (4250) → selected!**

### Prediction for new sample: Size = 1800
- 1800 > 1750 → Right node → $\hat{y} = 400$ (mean of samples 3, 4, 5)

---

## 10. Full Manual Training Example

### Dataset (6 rows)

| ID | X1: Rooms | X2: Age (yrs) | y: Price ($k) |
|---|---|---|---|
| 1 | 2 | 10 | 100 |
| 2 | 2 | 5 | 120 |
| 3 | 3 | 15 | 180 |
| 4 | 3 | 3 | 200 |
| 5 | 4 | 8 | 280 |
| 6 | 4 | 12 | 260 |

### Step 1: Root Node MSE

$$\bar{y}_{root} = \frac{100+120+180+200+280+260}{6} = 190$$

$$MSE_{root} = \frac{(100-190)^2+(120-190)^2+(180-190)^2+(200-190)^2+(280-190)^2+(260-190)^2}{6}$$
$$= \frac{8100+4900+100+100+8100+4900}{6} = \frac{26200}{6} = 4366.67$$

### Step 2: Try Split on Rooms ≤ 2

**Left (Rooms ≤ 2):** IDs 1, 2 → y = [100, 120] → $\bar{y} = 110$
$$MSE_{Left} = \frac{(100-110)^2+(120-110)^2}{2} = \frac{100+100}{2} = 100$$

**Right (Rooms > 2):** IDs 3, 4, 5, 6 → y = [180, 200, 280, 260] → $\bar{y} = 230$
$$MSE_{Right} = \frac{(180-230)^2+(200-230)^2+(280-230)^2+(260-230)^2}{4}$$
$$= \frac{2500+900+2500+900}{4} = 1700$$

$$Cost = \frac{2}{6}(100) + \frac{4}{6}(1700) = 33.33 + 1133.33 = 1166.67$$

### Step 3: Try Split on Age ≤ 8

**Left (Age ≤ 8):** IDs 2, 4, 5 → y = [120, 200, 280] → $\bar{y} = 200$
$$MSE_{Left} = \frac{(120-200)^2+(200-200)^2+(280-200)^2}{3} = \frac{6400+0+6400}{3} = 4266.67$$

**Right (Age > 8):** IDs 1, 3, 6 → y = [100, 180, 260] → $\bar{y} = 180$
$$MSE_{Right} = \frac{(100-180)^2+(180-180)^2+(260-180)^2}{3} = \frac{6400+0+6400}{3} = 4266.67$$

$$Cost = \frac{3}{6}(4266.67) + \frac{3}{6}(4266.67) = 4266.67$$

### Step 4: Choose Best Split

| Split | Cost |
|---|---|
| Rooms ≤ 2 | **1166.67 ✅ (best)** |
| Age ≤ 8 | 4266.67 |

**Best Split: Rooms ≤ 2**

### Step 5: Recurse on Right Node (Rooms > 2)

Samples: IDs 3, 4, 5, 6 → y = [180, 200, 280, 260]

Try Rooms ≤ 3:
- Left (Rooms = 3): IDs 3, 4 → y = [180, 200] → $\bar{y} = 190$, $MSE = 100$
- Right (Rooms = 4): IDs 5, 6 → y = [280, 260] → $\bar{y} = 270$, $MSE = 100$
- $Cost = \frac{2}{4}(100) + \frac{2}{4}(100) = 100$

**Split on Rooms ≤ 3!**

### Final Tree (Depth = 2):

```
Rooms ≤ 2?
├── Yes → Predict: 110
└── No → Rooms ≤ 3?
         ├── Yes → Predict: 190
         └── No  → Predict: 270
```

**Prediction for new house: Rooms = 3, Age = 7**
- Rooms ≤ 2? No → Rooms ≤ 3? Yes → **Predict: $190k**

---

## 11. Parameters (Learned Values)

| Parameter | Meaning |
|---|---|
| Split Feature | Which feature is used at each internal node |
| Split Threshold | The value that divides left (≤) and right (>) branches |
| Leaf Prediction Value | Mean of target values of training samples in that leaf |
| Node MSE | Variance of target values at each node |
| Node Sample Count | Number of training samples reaching each node |
| Node Weighted MSE | Weighted impurity contribution (used in feature importance) |

All of these are **learned automatically** during training — you do not set them manually.

---

## 12. Hyperparameters

| Hyperparameter | Default | Purpose |
|---|---|---|
| `max_depth` | None | Maximum depth of the tree |
| `min_samples_split` | 2 | Minimum samples needed to split a node |
| `min_samples_leaf` | 1 | Minimum samples required in each leaf |
| `max_features` | None | Number of features to consider at each split |
| `criterion` | "squared_error" | Split metric: "squared_error", "friedman_mse", "absolute_error", "poisson" |
| `max_leaf_nodes` | None | Limit on total number of leaf nodes |
| `min_impurity_decrease` | 0.0 | Minimum MSE reduction required to split |
| `ccp_alpha` | 0.0 | Complexity parameter for cost-complexity pruning |

---

## 13. Why Each Hyperparameter Exists

### `max_depth`
**Problem:** Unconstrained trees grow until every leaf holds 1 sample — MSE becomes 0 on training data. This is perfect memorization, not learning.  
**Solution:** `max_depth` stops the tree at a fixed level. Start with 3–6 for most problems.  
**Effect:** Deeper → more complex model → lower bias, higher variance → overfitting risk increases.

### `min_samples_split`
**Problem:** A node with 2 samples can still be split, creating hyper-specific rules.  
**Solution:** Only split a node with at least `min_samples_split` samples.  
**Practical use:** Set to 10–20 for medium datasets.

### `min_samples_leaf`
**Problem:** Leaf nodes with very few samples produce unstable mean estimates.  
**Solution:** Require at least `min_samples_leaf` samples per leaf.  
**Practical use:** More powerful than `min_samples_split` for controlling leaf stability.

### `criterion`
**Problem:** Different error metrics make sense for different regression tasks.  
- `squared_error` (MSE): Standard; penalizes large errors heavily.
- `absolute_error` (MAE): More robust to outliers.
- `friedman_mse`: Weighted MSE that often performs better in practice.
- `poisson`: For count data (Poisson-distributed targets).  
**Solution:** Choose based on the distribution of your target variable.

### `ccp_alpha`
**Problem:** Fully grown trees overfit.  
**Solution:** Post-pruning penalty — higher `ccp_alpha` removes more branches.  
Tune with cross-validation to find the optimal value.

---

## 14. Assumptions

Decision Tree Regressor makes very few formal assumptions:

1. **No linearity assumption:** Can model any non-linear relationship.
2. **No normality assumption:** Target doesn't need to be normally distributed.
3. **No homoscedasticity assumption:** Variance of errors can differ across the feature space.
4. **Monotonic transformations don't matter:** Since splits are threshold-based, scaling features has no effect.
5. **Implicit assumption:** The training data is a representative sample of the problem.
6. **Piecewise constant assumption:** The model implicitly assumes the true function is well-approximated by a step function (constant within each region).

---

## 15. Data Requirements

| Requirement | Needed? | Notes |
|---|---|---|
| Feature Scaling (Normalization) | ❌ No | Threshold-based splits are scale-invariant |
| Encoding Categorical Features | ✅ Yes | scikit-learn requires numerical input |
| Handling Missing Values | ✅ Yes | Must impute; scikit-learn raises error |
| Large Dataset | Recommended | Small datasets → high variance predictions |
| Outlier Removal | ⚠️ Recommended | Outliers can become isolated leaves with extreme predicted values |
| Target Distribution | No restriction | Works with any continuous distribution |

---

## 16. Complexity

### Training
$$O(n \cdot m \cdot \log n)$$

For each of $m$ features, we evaluate $O(n)$ thresholds at each of $O(\log n)$ tree levels (balanced tree). For unbalanced trees, worst case is $O(n^2 \cdot m)$.

### Prediction
$$O(depth)$$

Following a single path from root to leaf. For a balanced tree: $O(\log n)$.

### Memory
$$O(n \cdot m + n_{leaves})$$

Storing the training data for split computation, plus the tree structure. In the worst case (fully grown), there are $O(n)$ leaf nodes.

---

## 17. Decision Boundary

Unlike classifiers where we talk about decision boundaries, for regressors we talk about the **partitioning of the feature space** into rectangular regions, each with a constant predicted value.

```
Feature 2 (Age)
    |
 20 |  ŷ=100  |  ŷ=180  |
    |         |         |
 10 |  ŷ=100  |  ŷ=200  |
    |_________|_________|_________
    0         2         4      Feature 1 (Rooms)
```

**Key characteristics:**
- The predicted function is a **piecewise constant step function**.
- Boundaries are always **axis-aligned** (horizontal or vertical).
- Each rectangle = one leaf node.
- The more leaves, the finer the step function approximation.
- **Cannot extrapolate:** Predictions outside the range of training data are capped at the nearest leaf mean.

---

## 18. Overfitting & Underfitting

### Overfitting (most common problem)
A fully grown tree creates a separate leaf for every training sample (or small groups), so each leaf's prediction is just the value(s) of those specific training points.

**Signs:**
- Training MSE ≈ 0
- Test MSE is very high (orders of magnitude larger)
- The prediction function looks like a jagged staircase with many tiny steps

**Root cause:** Deep trees memorize the noise in training data as "rules."

### Underfitting
A tree that is too shallow (e.g., max_depth=1) fails to capture the structure of the data.

**Signs:**
- Both training and test MSE are high
- The prediction function is flat or has only 1–2 steps

**Fix:** Increase `max_depth` or decrease `min_samples_split`.

### Bias-Variance Tradeoff in Decision Trees
- **Deep tree:** Low bias (fits training data well), high variance (sensitive to data changes)
- **Shallow tree:** High bias (underfits), low variance (stable predictions)
- **Optimal:** Found via cross-validation on `max_depth`

---

## 19. Regularization

### Pre-Pruning (Early Stopping)
Prevents the tree from becoming too complex during training:

| Method | Effect |
|---|---|
| `max_depth` | Hard cap on tree depth |
| `min_samples_split` | Requires minimum samples before splitting |
| `min_samples_leaf` | Ensures each leaf has enough support |
| `min_impurity_decrease` | Only splits if MSE reduction exceeds threshold |
| `max_leaf_nodes` | Caps total number of leaves |

### Post-Pruning: Cost-Complexity Pruning

After building a full tree, prune branches by minimizing:

$$R_\alpha(T) = \frac{1}{n}\sum_{leaves} MSE_{leaf} \cdot n_{leaf} + \alpha \cdot |leaves|$$

- $\alpha$ (ccp_alpha) controls the trade-off between accuracy and simplicity
- Higher $\alpha$ → more branches pruned → simpler, more regularized tree
- Find optimal $\alpha$ using cross-validation

**scikit-learn implementation:**
```python
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas  # test these with cross-validation
```

### L1 / L2 Regularization
Not directly applicable to Decision Trees. These concepts apply to linear and neural network models.

---

## 20. Feature Importance

Feature importance in Decision Tree Regressor is computed as the **weighted variance reduction** across all nodes that split on a feature:

$$Importance(f) = \sum_{\text{nodes splitting on } f} \frac{n_{node}}{n_{total}} \cdot \left(MSE_{parent} - \frac{n_{left}}{n_{node}} MSE_{left} - \frac{n_{right}}{n_{node}} MSE_{right}\right)$$

**Normalized so all importances sum to 1.0.**

**Access in scikit-learn:**
```python
model.feature_importances_
```

**Limitations:**
- Biased toward features with many unique values (continuous features dominate categorical ones)
- Correlated features split importance between themselves
- Does not indicate the direction of the effect (positive or negative)

**Better alternatives:** SHAP values, partial dependence plots (for directional insight).

---

## 21. Advantages

- **Highly interpretable:** Predictions can be traced through explicit rules
- **No feature scaling required:** Threshold-based splits are scale-invariant
- **Captures non-linear relationships:** Arbitrary step function approximation
- **Fast prediction:** $O(\log n)$ per sample once trained
- **Handles mixed feature types:** Works with both numerical and categorical inputs
- **No distributional assumptions:** Non-parametric model
- **Handles outliers reasonably:** Outliers may end up in their own leaf but don't corrupt the whole model
- **Good base learner:** Forms the foundation of Gradient Boosted Trees and Random Forest

---

## 22. Limitations

- **High variance (unstable):** Small dataset changes can produce very different trees
- **Prone to overfitting:** Especially without depth constraints
- **Piecewise constant predictions:** Cannot capture smooth, continuous trends natively
- **Cannot extrapolate:** Predictions outside the training range are bounded by training extremes
- **Biased toward high-cardinality features:** Features with many unique values dominate splits
- **Greedy algorithm:** Does not guarantee globally optimal tree
- **Poor with linear relationships:** Requires many splits to approximate a simple line

---

## 23. Failure Cases

| Scenario | Why It Fails | Solution |
|---|---|---|
| Smooth, continuous target | Step-function approximation is rough | Use linear regression or SVR |
| Target outside training range | Cannot extrapolate; predictions capped | Use linear models for extrapolation |
| Many outliers in target | Outlier leaves get extreme mean values | Use MAE criterion or remove outliers |
| Linear relationship | Requires many splits to approximate a line | Use linear regression instead |
| Small dataset (< 50 samples) | Overfits severely | Use cross-validation + strong pruning |
| High-dimensional data (many features) | Unstable; correlated features confuse splits | Use Random Forest or feature selection |

---

## 24. Edge Cases

| Situation | Behavior |
|---|---|
| **Tiny dataset (< 20 samples)** | Creates a tree with 1 sample per leaf; MSE = 0 on training, terrible on test |
| **Huge dataset (millions of rows)** | Training is slow; use `max_features='sqrt'` and `min_samples_leaf=50+` |
| **Features > Samples** | Perfect overfitting; aggressive pruning or ensemble methods required |
| **Target has outliers (e.g., 1M vs normal 100k)** | Outlier becomes isolated leaf; all other predictions are pulled toward outlier mean if in same region |
| **Missing values** | scikit-learn raises ValueError; must impute beforehand |
| **Constant target (all y equal)** | MSE = 0 everywhere; tree is just one node predicting that value |
| **Duplicate features** | Model handles it but wastes time computing redundant splits |
| **Target is count data** | Use `criterion='poisson'` for better splits |

---

## 25. Evaluation Metrics

### Regression Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | Average absolute error; same units as target; robust to outliers |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Penalizes large errors heavily; sensitive to outliers |
| **RMSE** | $\sqrt{MSE}$ | Same units as target; easier to interpret than MSE |
| **R² Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained; 1.0 = perfect, 0 = no better than mean |
| **MAPE** | $\frac{100}{n}\sum \left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Percentage error; useful for comparing across different scales |
| **Explained Variance** | $1 - \frac{Var(y - \hat{y})}{Var(y)}$ | Similar to R² but doesn't penalize systematic bias |

### Formulas in Detail

**R² Score:**
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
- R² = 1: Perfect prediction
- R² = 0: Model predicts the mean — no better than a trivial baseline
- R² < 0: Worse than predicting the mean (very bad model)

---

## 26. Comparison with Similar Models

| Feature | Decision Tree Regressor | Random Forest Regressor | Linear Regression | Gradient Boosting |
|---|---|---|---|---|
| Interpretability | ✅ High | ❌ Low | ✅ Very High | ❌ Low |
| Non-linearity | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| Overfitting Risk | ❌ High | ✅ Low | ✅ Low | ⚠️ Medium |
| Extrapolation | ❌ No | ❌ No | ✅ Yes | ❌ No |
| Training Speed | ✅ Fast | ❌ Slower | ✅ Very Fast | ❌ Slow |
| Feature Scaling | ❌ Not needed | ❌ Not needed | ✅ Needed | ❌ Not needed |
| Handles Outliers | ⚠️ Moderate | ⚠️ Moderate | ❌ Sensitive | ⚠️ Moderate |
| Smooth Predictions | ❌ Step function | ⚠️ Smoother | ✅ Smooth line | ✅ Smooth |
| Base for Ensembles | ✅ Yes | N/A | ❌ No | ✅ Yes |

---

## 27. Real-World Applications

| Domain | Application |
|---|---|
| **Real Estate** | Predicting house prices from area, location, age, and features |
| **Finance** | Stock price prediction, risk assessment, loan amount estimation |
| **Healthcare** | Predicting patient recovery time, drug dosage estimation |
| **Energy** | Forecasting electricity consumption based on temperature and time |
| **Logistics** | Estimating delivery times based on route, weight, and weather |
| **Manufacturing** | Predicting product defect rates, machine maintenance intervals |
| **Marketing** | Predicting customer lifetime value, campaign ROI |
| **Agriculture** | Crop yield prediction based on soil, weather, and input features |

---

## 28. Scikit-Learn Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Prepare Data
# ----------------------------
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Note: No scaling needed for Decision Trees!

# ----------------------------
# 2. Train Model
# ----------------------------
model = DecisionTreeRegressor(
    criterion='squared_error',   # or 'friedman_mse', 'absolute_error', 'poisson'
    max_depth=6,                 # Limit depth to prevent overfitting
    min_samples_split=20,        # Min samples needed to split
    min_samples_leaf=10,         # Min samples in each leaf
    random_state=42
)
model.fit(X_train, y_train)

# ----------------------------
# 3. Predict
# ----------------------------
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ----------------------------
# 4. Evaluate
# ----------------------------
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print("=== Test Set Performance ===")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Check overfitting
r2_train = r2_score(y_train, y_pred_train)
print(f"\nTrain R²: {r2_train:.4f}")
print(f"Test R²:  {r2:.4f}")
print(f"Overfit gap: {r2_train - r2:.4f}")

# ----------------------------
# 5. Feature Importance
# ----------------------------
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n=== Feature Importances ===")
for i in range(len(feature_names)):
    print(f"  {feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")

# Plot
plt.figure(figsize=(10, 5))
plt.bar(range(len(feature_names)),
        importances[sorted_idx],
        tick_label=[feature_names[i] for i in sorted_idx])
plt.title("Feature Importances")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.show()

# ----------------------------
# 6. Visualize Tree
# ----------------------------
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, filled=True, max_depth=3,
          rounded=True, fontsize=10)
plt.title("Decision Tree Regressor (max_depth=3 shown)")
plt.savefig("decision_tree_regressor.png", dpi=150, bbox_inches='tight')
plt.show()

# Print rules as text
rules = export_text(model, feature_names=list(feature_names))
print("\nDecision Rules (first 30 lines):")
print('\n'.join(rules.split('\n')[:30]))

# ----------------------------
# 7. Visualize Predicted vs Actual
# ----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"Predicted vs Actual (R² = {r2:.3f})")
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=150)
plt.show()

# ----------------------------
# 8. Hyperparameter Tuning
# ----------------------------
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20, 50],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
}

grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"\nBest Parameters: {grid_search.best_params_}")
best_mse = -grid_search.best_score_
print(f"Best CV RMSE: {np.sqrt(best_mse):.4f}")

# ----------------------------
# 9. Cost-Complexity Pruning
# ----------------------------
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[::5]  # Sample every 5th value for speed

train_scores, test_scores = [], []
for alpha in ccp_alphas:
    reg = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42)
    reg.fit(X_train, y_train)
    train_scores.append(r2_score(y_train, reg.predict(X_train)))
    test_scores.append(r2_score(y_test, reg.predict(X_test)))

optimal_idx = np.argmax(test_scores)
optimal_alpha = ccp_alphas[optimal_idx]
print(f"\nOptimal ccp_alpha: {optimal_alpha:.6f}")
print(f"Test R² at optimal alpha: {test_scores[optimal_idx]:.4f}")

# Final pruned model
pruned_model = DecisionTreeRegressor(ccp_alpha=optimal_alpha, random_state=42)
pruned_model.fit(X_train, y_train)
print(f"Tree depth after pruning: {pruned_model.get_depth()}")
print(f"Number of leaves: {pruned_model.get_n_leaves()}")
```

---

## 29. Interview Questions

### Basic

1. **How does Decision Tree Regressor differ from Classifier?**  
   The Classifier assigns a class label (majority vote) to each leaf. The Regressor assigns a continuous value (mean of target values) to each leaf. The split criterion also differs: Classifier uses Gini/Entropy, Regressor uses MSE/MAE.

2. **What does a leaf node predict in a Decision Tree Regressor?**  
   The mean of the target values of all training samples that fell into that leaf during training.

3. **Why is MSE used as the split criterion?**  
   MSE measures the variance of target values in a node. Minimizing MSE after a split means we're creating groups where each group's values are as homogeneous as possible, making the mean a good prediction.

4. **Can Decision Tree Regressor extrapolate?**  
   No. It predicts the mean of training samples in a leaf. For inputs outside the training range, it returns the value of the nearest leaf — no extrapolation beyond training extremes.

### Intermediate

5. **Why does a fully grown Decision Tree Regressor overfit?**  
   With enough depth, it creates one leaf per training sample with MSE = 0 on training. The model memorizes the exact training values rather than learning a generalizable pattern.

6. **What is the difference between `squared_error` and `friedman_mse`?**  
   Both measure mean squared error, but `friedman_mse` adds an improvement score that weights the split based on the number of samples and the estimated mean improvement — often performs better in practice. Named after Jerome Friedman who proposed this variant.

7. **How do you handle outliers in the target variable?**  
   Use `criterion='absolute_error'` (MAE criterion) instead of `squared_error`. Since MAE minimizes toward the median rather than the mean, it is more robust to outliers.

8. **What is the bias-variance trade-off in Decision Trees?**  
   Deep trees have low bias (fit training data well) but high variance (sensitive to small data changes). Shallow trees have high bias but low variance. Optimal depth found via cross-validation.

9. **Why can't Decision Trees extrapolate?**  
   They partition the feature space into rectangular regions based on training data ranges. No rule exists for inputs outside those ranges; predictions are capped at the extremes of training data.

### Advanced

10. **What is the time complexity of training and why?**  
    $O(n \cdot m \cdot \log n)$: At each of $O(\log n)$ levels, we evaluate $O(n)$ thresholds for each of $m$ features. This is for a balanced tree; unbalanced worst case is $O(n^2 \cdot m)$.

11. **How is feature importance computed in the Regressor?**  
    Weighted MSE reduction across all nodes that use a given feature: $\sum_{nodes\ on\ f} \frac{n_{node}}{n_{total}} \cdot (MSE_{before} - MSE_{after})$. This captures how much each feature reduced the prediction error across the entire tree.

12. **Decision Tree Regressor vs Gradient Boosted Trees — when to use each?**  
    Decision Tree: when interpretability is critical, quick baseline needed, or as exploratory analysis.  
    Gradient Boosting: when predictive performance is the primary goal and black-box is acceptable. Gradient Boosting trains many shallow trees sequentially to correct previous errors, achieving much lower bias.

13. **What is `ccp_alpha` and how do you tune it?**  
    Cost-Complexity Pruning alpha. Higher values prune more branches. Tune by: (1) compute the pruning path with `cost_complexity_pruning_path()`, (2) train trees with each alpha value, (3) evaluate on validation set, (4) choose alpha that maximizes test R² (or minimizes test RMSE).

14. **Why is Decision Tree Regressor called a piecewise constant approximator?**  
    Because the predicted function is a step function — it assigns a constant value (the leaf mean) to each rectangular region of the feature space. It approximates any smooth function as a collection of horizontal plateaus.

15. **When would you use `criterion='poisson'` over `criterion='squared_error'`?**  
    When the target variable is count data (non-negative integers like number of accidents, page views, or insurance claims). Poisson criterion uses the Poisson deviance, which is more appropriate for count distributions and naturally handles the mean-variance relationship in count data.
