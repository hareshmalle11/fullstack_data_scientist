import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase URL or Key is missing. Check your .env file.")

sb: Client = create_client(url, key)

def get_all_products():
    resp = sb.table("products").select("*").execute()
    return resp.data

def get_product_by_id(prod_id):
    resp = sb.table("products").select("*").eq("prod_id", prod_id).execute()
    return resp.data

if __name__ == "__main__":
    choice = input("Do you want all products or a specific one? (all/id): ").strip().lower()
    
    if choice == "all":
        products = get_all_products()
        print("Products:", products)
    elif choice == "id":
        prod_id = int(input("Enter product id: ").strip())
        product = get_product_by_id(prod_id)
        print("Product:", product)
    else:
        print("Invalid choice")
