'''5. Hospital Patient Management

Scenario:

A hospital needs a system to manage patient records. Write a program to store patient data, including **name, age, and disease**, and allow the admin to search for patients by disease.

Requirements:

- Store patient records in a list of dictionaries.

- Allow searching for patients based on disease.

- Optional: Use a `Patient` class to manage records.

Input Example:

patients = [

    {"Name": "Alice", "Age": 30, "Disease": "Flu"},

    {"Name": "Bob", "Age": 45, "Disease": "Diabetes"},

    {"Name": "Charlie", "Age": 35, "Disease": "Flu"}

]

search_disease = "Flu"

Expected Output:

Patients with Flu: ["Alice", "Charlie"]'''

class P:
    def __init__(self, n, a, d):
        self.n = n
        self.a = a
        self.d = d

def find_by_illness(rec, ill):
    return [p["Name"] for p in rec if p["Disease"].lower() == ill.lower()]

if __name__ == "__main__":
    rec = [
        {"Name": "Alice", "Age": 30, "Disease": "Flu"},
        {"Name": "Bob", "Age": 45, "Disease": "Diabetes"},
        {"Name": "Charlie", "Age": 35, "Disease": "Flu"},
    ]
    ill = "Flu"
    res = find_by_illness(rec, ill)
    print(f"Patients with {ill}: {res}")