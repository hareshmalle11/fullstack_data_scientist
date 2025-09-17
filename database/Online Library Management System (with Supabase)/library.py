import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
sb: Client = create_client(url, key)

def add_member(member_id,name, email):
    payload = {"member_id":member_id,"name": name, "email": email}
    resp = sb.table("members").insert(payload).execute()
    return resp.data
def add_book(title, author, category=None, stock=1, book_id=None):
    payload = {"title": title, "author": author, "category": category, "stock": stock}
    resp = sb.table("books").insert(payload).execute()
    return resp.data
def update_book_stock(book_id, new_stock):
    resp = sb.table("books").update({"stock": new_stock}).eq("book_id", book_id).execute()
    return resp.data
def update_member_email(member_id, new_email):
    resp = sb.table("members").update({"email": new_email}).eq("member_id", member_id).execute()
    return resp.data
def delete_member(member_id):
    active = sb.table("borrow_records").select("*").eq("member_id", member_id).is_("return_date", None).execute()
    if active.data:
        print("Cannot delete member: they have active borrowed books.")
        return None
    resp = sb.table("members").delete().eq("member_id", member_id).execute()
    return resp.data
def delete_book(book_id):
    active = sb.table("borrow_records").select("*").eq("book_id", book_id).is_("return_date", None).execute()
    if active.data:
        print("Cannot delete book: currently borrowed.")
        return None
    resp = sb.table("books").delete().eq("book_id", book_id).execute()
    return resp.data
def list_books():
    resp = sb.table("books").select("*").execute()
    return resp.data
def list_members():
    resp = sb.table("members").select("*").execute()
    return resp.data
def borrow_book(member_id, book_id):
    book = sb.table("books").select("*").eq("book_id", book_id).execute()
    if not book.data:
        print("Book not found")
        return
    if book.data[0]['stock'] <= 0:
        print("Book not available")
        return
    sb.table("books").update({"stock": book.data[0]['stock'] - 1}).eq("book_id", book_id).execute()
    resp = sb.table("borrow_records").insert({"member_id": member_id, "book_id": book_id}).execute()
    return resp.data
def return_book(member_id, book_id):

    record = sb.table("borrow_records").select("*").eq("member_id", member_id).eq("book_id", book_id).is_("return_date", None).execute()
    if not record.data:
        print("No active borrow record found")
        return

    sb.table("borrow_records").update({"return_date": "NOW()"}).eq("record_id", record.data[0]['record_id']).execute()

    book = sb.table("books").select("*").eq("book_id", book_id).execute()
    sb.table("books").update({"stock": book.data[0]['stock'] + 1}).eq("book_id", book_id).execute()
    return f"Book {book_id} returned by member {member_id}"

def main():
    while True:
        print("\n--- Library Management ---")
        print("1. Add Member")
        print("2. Add Book")
        print("3. Update Book Stock")
        print("4. Update Member Email")
        print("5. Delete Member")
        print("6. Delete Book")
        print("7. List Members")
        print("8. List Books")
        print("9. Borrow Book")
        print("10. Return Book")
        print("0. Exit")

        choice = int(input("Enter choice: "))

        if choice == 1:
            name = input("Member name: ").strip()
            email = input("Member email: ").strip()
            member_id = int(input("Member ID (optional): "))
            print(add_member( member_id,name, email))

        elif choice == 2:
            title = input("Book title: ").strip()
            author = input("Author: ").strip()
            category = input("Category (optional): ").strip()
            stock = int(input("Stock quantity: ").strip() or 1)
            id_input = input("Book ID (optional): ").strip()
            book_id = int(id_input) if id_input else None
            print(add_book(title, author, category, stock, book_id))

        elif choice == 3:
            book_id = int(input("Book ID to update stock: ").strip())
            new_stock = int(input("New stock: ").strip())
            print(update_book_stock(book_id, new_stock))

        elif choice == 4:
            member_id = int(input("Member ID to update email: ").strip())
            new_email = input("New email: ").strip()
            print(update_member_email(member_id, new_email))

        elif choice == 5:
            member_id = int(input("Member ID to delete: ").strip())
            print(delete_member(member_id))

        elif choice == 6:
            book_id = int(input("Book ID to delete: ").strip())
            print(delete_book(book_id))

        elif choice == 7:
            members = list_members()
            for m in members:
                print(m)

        elif choice == 8:
            books = list_books()
            for b in books:
                print(b)

        elif choice == 9:
            member_id = int(input("Member ID: ").strip())
            book_id = int(input("Book ID: ").strip())
            print(borrow_book(member_id, book_id))

        elif choice == 10:
            member_id = int(input("Member ID: ").strip())
            book_id = int(input("Book ID: ").strip())
            print(return_book(member_id, book_id))

        elif choice == 0:
            print("Exiting...")
            break


if __name__ == "__main__":
    main()
