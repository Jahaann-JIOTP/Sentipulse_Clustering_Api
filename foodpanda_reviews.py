import requests
from pymongo import MongoClient
from datetime import datetime
import time
import traceback

# === MongoDB Setup ===
client = MongoClient("mongodb://admin:cisco123@13.234.241.103:27017/sentipulse?authSource=admin&readPreference=primary&ssl=false")
db = client["sentipulse"]
collection = db["ginyaki_foodpanda_reviews"]
log_collection = db["ginyaki_foodpanda_logs"]

# === Log Start Time ===
log_start_time = datetime.utcnow()

# === Get latest stored review datetime ===
latest_doc = collection.find_one(
    {"date": {"$exists": True}},
    sort=[("date", -1)]
)
latest_date = latest_doc["date"] if latest_doc else None
print(f"📅 Latest stored review date: {latest_date}")

# === Base API Setup ===
BASE_URL = "https://reviews-api-pk.fd-api.com/reviews/vendor/t2vx"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

params = {
    "global_entity_id": "FP_PK",
    "limit": 30,
    "created_at": "desc",
    "has_dish": "true"
}

next_page_key = None
total_fetched = 0
page = 1
error_message = None

try:
    while True:
        if next_page_key:
            params["nextPageKey"] = next_page_key
        elif "nextPageKey" in params:
            del params["nextPageKey"]

        print(f"\n🔄 Fetching page {page}...")

        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            error_message = f"HTTP Error {response.status_code}"
            print(f"❌ Request failed at page {page}: {error_message}")
            break

        json_data = response.json()
        reviews = json_data.get("data", [])
        next_page_key = json_data.get("pageKey")
        print(f"▶️ nextPageKey for next page: {next_page_key}")

        if not reviews:
            print("✅ No more reviews found in this page.")
            break

        for item in reviews:
            try:
                created_at = item.get("createdAt")
                created_at_dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")

                # Skip if not newer than latest
                if latest_date and created_at_dt <= latest_date:
                    print(f"⏭️ Skipped review ({created_at_dt}) - already stored")
                    continue

                # Extract details
                reviewer = item.get("reviewerName")
                review_text = item.get("text")
                ratings_list = item.get("ratings", [])
                ratings_map = {r["topic"]: r["score"] for r in ratings_list}
                overall_rating = ratings_map.get("overall")
                restaurant_food_rating = ratings_map.get("restaurant_food")
                rider_rating = ratings_map.get("rider")

                product_variations = item.get("productVariations", [])
                if product_variations:
                    product_info = product_variations[0]
                    product_title = product_info.get("product", {}).get("title", {}).get("en_PK", "")
                    product_price = product_info.get("unitPrice", "")
                else:
                    product_title = ""
                    product_price = ""

                doc = {
                    "user": reviewer,
                    "text": review_text,
                    "overall_rating": overall_rating,
                    "restaurant_food_rating": restaurant_food_rating,
                    "rider_rating": rider_rating,
                    "date": created_at_dt,
                    "product": product_title,
                    "price_per_person": product_price,
                    "restaurant": "Ginyaki",
                    "source": "Foodpanda"
                }

                collection.insert_one(doc)
                total_fetched += 1
                print(f"✅ Saved review #{total_fetched} ({created_at_dt})")

            except Exception as e:
                print(f"❌ Skipped one review due to error: {e}")

        if not next_page_key:
            print("✅ Reached end of pages.")
            break

        page += 1
        time.sleep(1)

except Exception as e:
    error_message = traceback.format_exc()
    print(f"❌ Script crashed due to: {error_message}")

# === Log Run Details ===
log_entry = {
    "timestamp": log_start_time,
    "latest_stored_review_date": str(latest_date) if latest_date else None,
    "total_reviews_fetched": total_fetched,
    "status": "success" if not error_message else "failure",
    "error_message": error_message if error_message else None
}
log_collection.insert_one(log_entry)
print("📝 Run log saved to MongoDB (ginyaki_reviews_all_logs).")

print(f"\n🎉 DONE! Total {total_fetched} new reviews saved to MongoDB.")
