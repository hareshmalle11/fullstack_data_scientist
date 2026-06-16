'''2. Restaurant Menu Management

Scenario:

You are managing a restaurant's menu. Write a program to update the menu by adding or removing items and allow users to check if a particular item is available.

Requirements:

- Use functions for adding, removing, and checking menu items.

- Handle cases where the item to be removed does not exist.

Input Example:

initial_menu = ["Pizza", "Burger", "Pasta", "Salad"]

add_item = "Tacos"

remove_item = "Salad"

check_item = "Pizza"

Expected Output:

Updated menu: ["Pizza", "Burger", "Pasta", "Tacos"]

Availability: "Pizza is available"'''

def add(food, lst):
    if food not in lst:
        lst.append(food)
    return lst

def remove(food, lst):
    if food in lst:
        lst.remove(food)
    return lst

def check(food, lst):
    if food in lst:
        return f"{food} is available"
    return f"{food} is not available"

if __name__ == "__main__":
    m = ["Pizza", "Burger", "Pasta", "Salad"]
    m = add("Tacos", m)
    m = remove("Salad", m)
    print("Updated menu:", m)
    print("Availability:", check("Pizza", m))
    print("Availability:", check("Sushi", m))