import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")



sb: Client = create_client(url, key)

def update_price(prod_id, new_price):
    resp = sb.table("products").update({"price": new_price}).eq("prod_id", prod_id).execute()
    return resp.data

if __name__ == "__main__":
    prod_id = int(input("Enter product id to update: ").strip())
    new_price = int(input("Enter new price: ").strip())
    
    updated = update_price(prod_id, new_price)
    print("Updated:", updated)
