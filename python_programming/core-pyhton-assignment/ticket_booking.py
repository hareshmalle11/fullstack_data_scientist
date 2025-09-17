'''4. Movie Ticket Booking System

Scenario:

A cinema hall wants to manage ticket bookings. Write a program to keep track of **available** and **booked seats**. Allow users to **book** or **cancel** a seat.

Requirements:

- Use functions to handle seat booking and cancellation.

- Prevent booking an already booked seat.

Input Example:

total_seats = 10

booked_seats = [2, 5, 7]

book_seat = 3

cancel_seat = 5

Expected Output:

Available seats: [1, 4, 6, 8, 9, 10]'''

def book(bkd, s, max_seats):
    if s in bkd:
        print(f"Seat {s} is already booked.")
    elif 1 <= s <= max_seats:
        bkd.append(s)
    else:
        print("Invalid seat number.")
    return bkd

def cancel(bkd, s):
    if s in bkd:
        bkd.remove(s)
    else:
        print(f"Seat {s} is not booked.")
    return bkd

def free_seats(max_seats, bkd):
    return [s for s in range(1, max_seats + 1) if s not in bkd]

if __name__ == "__main__":
    max_seats = 10
    bkd = [2, 5, 7]
    bkd = book(bkd, 3, max_seats)
    bkd = cancel(bkd, 5)
    print("Available seats:", free_seats(max_seats, bkd))