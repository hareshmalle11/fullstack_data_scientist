'''6. Customer Feedback Analysis

Scenario:

A company collects customer feedback in the form of ratings (1-5). Write a program to calculate the **percentage of positive feedback** (4 or 5).

Requirements:

- Use a function to calculate the positive feedback percentage.

- Handle cases where no ratings are available.

Input Example:

```python

ratings = [5, 4, 3, 5, 2, 4, 1, 5]

```

Expected Output:

```

Positive Feedback: 62.5%

```'''
def pos_pct(rates):
    if not rates:
        return 0
    pos = sum(1 for r in rates if r >= 4)
    return (pos / len(rates)) * 100

if __name__ == "__main__":
    rates = [5, 4, 3, 5, 2, 4, 1, 5]
    pct = pos_pct(rates)
    print(f"Positive Feedback: {pct:.1f}%")
