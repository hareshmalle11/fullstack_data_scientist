import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase URL or Key is missing. Check your .env file.")

sb: Client = create_client(url, key)

def delete_product(prod_id):
    resp = sb.table("products").delete().eq("prod_id", prod_id).execute()
    return resp.data

if __name__ == "__main__":
    prod_id = int(input("Enter product id to delete: ").strip())
    deleted = delete_product(prod_id)
    print("Deleted:", deleted)
