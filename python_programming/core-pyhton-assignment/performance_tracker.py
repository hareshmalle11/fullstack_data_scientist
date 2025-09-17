'''3. Classroom Performance Tracker

Scenario:A teacher wants to track student performance. Write a program to calculate the **average marks** of students and identify the **top performer**.

Requirements:

- Use a function to calculate the average marks.

- Identify the student with the highest average.

- Optional: Implement a `Student` class to handle this logic.

Input Example:

students = {"John": [85, 78, 92], "Alice": [88, 79, 95], "Bob": [70, 75, 80]}

Expected Output:

Average Marks: {"John": 85, "Alice": 87.33, "Bob": 75}

Top Performer: "Alice"'''

class S:
    def __init__(self, n, m):
        self.n = n
        self.m = m
    def avg(self):
        return sum(self.m)/len(self.m) if self.m else 0

def calc_avg(studs):
    avg = {n: S(n, m).avg() for n, m in studs.items()}
    return avg

def top(avg):
    return max(avg, key=avg.get)

if __name__ == "__main__":
    studs = {"John": [85, 78, 92], "Alice": [88, 79, 95], "Bob": [70, 75, 80]}
    avg = calc_avg(studs)
    print("Average Marks:", avg)
    print("Top Performer:", top(avg))