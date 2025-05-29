from apify_client import ApifyClient
from pymongo import MongoClient
from datetime import datetime, timedelta
from time import sleep

# === Apify Setup ===
APIFY_TOKEN = "apify_api_0OGBAJ1vENTrcKHPIqZegVSOCsQU8I3XAkGi"
ACTOR_ID = "Xb8osYTtOjlsgI6k9"
client = ApifyClient(APIFY_TOKEN)

# === MongoDB Setup ===
mongo_client = MongoClient("mongodb://admin:cisco123@13.234.241.103:27017/sentipulse?authSource=admin&readPreference=primary&ssl=false")
db = mongo_client["sentipulse"]
collection = db["ginyaki_google_reviews"]
log_collection = db["ginyaki_google_logs"]

# === Step 1: Get latest stored review date ===
latest_doc = collection.find_one(
    {"date": {"$exists": True}},
    sort=[("date", -1)]
)
latest_date = datetime.fromisoformat(str(latest_doc["date"])) if latest_doc else None
print(f"📅 Latest stored review date: {latest_date}")
# === Step 2: Build run_input ===
run_input = {
    "startUrls": [
        {
            "url": "https://www.google.com/maps/search/ginyaki/@33.6363279,73.006127,12z/data=!3m1!4b1!5m1!1e1?entry=ttu&g_ep=EgoyMDI1MDUwNS4wIKXMDSoASAFQAw%3D%3D",
            "method": "GET"
        }
    ],
    "includeReviews": True,
    "language": "en",
    "sortReviewsBy": "oldest",
    "personalData": True
}

# ✅ Pass 'reviewsStartDate' only if latest_date exists
if latest_date:
    next_day = latest_date + timedelta(days=1)
    run_input["reviewsStartDate"] = next_day.strftime("%Y-%m-%d")

# === Step 3: Run the Apify actor ===
print("🚀 Starting Apify actor...")
run = client.actor(ACTOR_ID).call(run_input=run_input)
dataset_id = run["defaultDatasetId"]

# === Step 4: Fetch the results ===
print("📥 Fetching dataset results...")
sleep(5)
dataset_items = client.dataset(dataset_id).list_items().items
print(f"📊 Reviews returned: {len(dataset_items)}")

# === Step 5: Store New Reviews in MongoDB ===
new_count = 0
for review in dataset_items:
    date = review.get("publishedAtDate")
    try:
        date = datetime.fromisoformat(date) if date else None
    except:
        date = None
    reviewContext = review.get("reviewContext", {})  # ✅ Safe access
    reviewDetailedRating = review.get("reviewDetailedRating", {})  # ✅ Safe access


    review_data = {
        "place": review.get("title", "Unknown Place"),
        "user": review.get("name"),
        "rating": review.get("rating") or review.get("stars"),
        "overall_rating": review.get("totalScore"),  # ✅ Added this line
        "text": review.get("text"),
        "date": date,
        "review_url": review.get("reviewUrl"),
        "review_id": review.get("reviewId"),
        "reviewer_profile": review.get("reviewerUrl"),
        "likes_count": review.get("likesCount"),
        "location": review.get("address"),
        "stars": review.get("stars"),
        "reviews_count": review.get("reviewsCount"),
        "meal_type": reviewContext.get("Meal type"),  # ✅ New field
        "price_per_person": reviewContext.get("Price per person"),
        "service_type": reviewContext.get("Service"),
        "reviewerPhotoUrl": review.get("reviewerPhotoUrl"),
        "rating_food": reviewDetailedRating.get("Food"),
        "rating_service": reviewDetailedRating.get("Service"),
        "rating_atmosphere": reviewDetailedRating.get("Atmosphere"),
    }
    if (
        review_data["review_id"]
        and review_data["date"]
        and not collection.find_one({"review_id": review_data["review_id"]})
    ):
        collection.insert_one(review_data)
        new_count += 1

print(f"✅ {new_count} new reviews stored in MongoDB.")

# === Step 6: Save Run Log ===
log_entry = {
    "timestamp": datetime.utcnow(),
    "reviews_start_date": run_input.get("reviewsStartDate", "N/A"),
    "total_reviews_fetched": len(dataset_items),
    "new_reviews_inserted": new_count,
    "search_query": [url["url"] for url in run_input.get("startUrls", [])],
    "status": "success" if new_count >= 0 else "failed"
}
log_collection.insert_one(log_entry)
print("📝 Log saved to MongoDB (ginyaki_reviews_all_logs).")
