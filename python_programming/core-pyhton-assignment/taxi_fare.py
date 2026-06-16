'''7. Taxi Fare Calculation

Scenario:

A taxi service calculates fares based on the following rates: 

- **Base fare**: $50 

- **Distance fare**: $10/km 

Write a program to calculate the total fare for multiple trips.

Requirements:

- Use a function to calculate fare for each trip.

Input Example:

```python

trips = [5, 10, 3]  # Distances in km

```

Expected Output:

```

Trip 1: $100

Trip 2: $150

Trip 3: $80

Total Fare: $330

```

 

---'''

BASE = 50
PER_KM = 10

def calc_fare(dist):
    return BASE + (PER_KM * dist)

if __name__ == "__main__":
    trips = [5, 10, 3]
    total = 0
    for i, d in enumerate(trips, start=1):
        f = calc_fare(d)
        total += f
        print(f"Trip {i}: ${f}")
    print(f"Total Fare: ${total}")