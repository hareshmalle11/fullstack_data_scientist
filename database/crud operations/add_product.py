from supabase import create_client, Client
from dotenv import load_dotenv
import os


load_dotenv()  # must be called before using os.getenv

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase URL or Key is missing. Check your .env file.")

sb: Client = create_client(url, key)

 
def add_product(prod_id,name,price, stock):
    payload = {"name": name, "prod_id":prod_id, "price": price, "stock": stock}
    resp = sb.table("products").insert(payload).execute()
    return resp.data
 
if __name__ == "__main__":
    prod_id = input("Enter product id: ").strip()
    name = input("Enter product name: ").strip()
    price = float(input("Enter price: ").strip())
    stock = int(input("Enter stock: ").strip())
 
    created = add_product(prod_id,name, price, stock)
    print("Inserted:", created)