import os
import time
import pandas as pd
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from collections import defaultdict
from modules.utils import standardize_df, save_log_to_buffer, save_logs_to_mongodb
from modules.sentiment import run_sentiment_analysis_pipeline

def sentipulseRunThePipeline(
    source_db_name="sentipulse",        # ✅ Using the same DB for source
    output_db_name="sentipulse",        # ✅ Changed to same DB for output
    **kwargs
):
    load_dotenv()
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    db = client[source_db_name]
    output_db = client[output_db_name]
    os.makedirs(output_db_name, exist_ok=True)

    def ensure_indexes():
        """Ensure indexes are created in the output database."""
        for collection_name in output_db.list_collection_names():
            output_db[collection_name].create_index([("text", ASCENDING), ("date", ASCENDING)])

    ensure_indexes()

    try:
        collections = db.list_collection_names()
        save_log_to_buffer(f"📚 Found {len(collections)} collections in '{source_db_name}' DB", "info")
        print(f"\n📚 Found {len(collections)} collections in '{source_db_name}' DB")
    except Exception as e:
        save_log_to_buffer(f"❌ Failed to list MongoDB collections: {e}", "error")
        print(f"❌ Error: {e}")
        save_logs_to_mongodb()
        return

    grouped = defaultdict(list)

    for name in collections:
        # ✅ Only include collections with exactly 3 parts and ending in '_reviews'
        parts = name.split('_')
        if len(parts) == 3 and parts[-1].lower() == "reviews":
            restaurant, platform, _ = parts
            grouped[restaurant].append((platform, name))
        else:
            save_log_to_buffer(f"⚠️ Skipping non-review collection: {name}", "warning")

    for restaurant, platforms in grouped.items():
        start_time = time.time()
        save_log_to_buffer(f"🍽️ Processing restaurant: {restaurant}", "info")
        print(f"\n🍽️ Processing restaurant: {restaurant}")
        merged_df = pd.DataFrame()

        # ✅ Save to {restaurant}_sentimented_reviews instead of overwriting raw
        output_collection_name = f"{restaurant}_sentimented_output"
        output_collection = output_db[output_collection_name]

        existing_max_date = None
        if output_collection_name in output_db.list_collection_names():
            try:
                max_date_doc = output_collection.find({}, {"date": 1}).sort("date", -1).limit(1)
                for doc in max_date_doc:
                    existing_max_date = pd.to_datetime(doc.get("date"), errors='coerce')
            except Exception as e:
                save_log_to_buffer(f"⚠️ Error fetching max date for {restaurant}: {e}", "error")
                print(f"⚠️ Error fetching max date for {restaurant}: {e}")

        for platform, collection_name in platforms:
            try:
                query = {"date": {"$gt": existing_max_date}} if existing_max_date else {}
                documents = list(db[collection_name].find(query))
                print(f"   📥 {len(documents)} new rows from: {collection_name}")
                save_log_to_buffer(f"📥 Read {len(documents)} from {collection_name}", "info")
                if not documents:
                    continue

                df = pd.DataFrame(documents)
                df['sent_restaurant'] = restaurant
                df['sent_platform'] = platform
                print(f"    🔄 Standardizing data for: {platform}")
                df = standardize_df(df,platform)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce', utc=True)

            except Exception as e:
                save_log_to_buffer(f"❌ Error in '{collection_name}': {e}", "error")
                print(f"❌ Error in '{collection_name}': {e}")
                save_logs_to_mongodb()

        if merged_df.empty:
            save_log_to_buffer(f"🟡 No new reviews for '{restaurant}', skipping.", "info")
            print(f"🟡 No new reviews for '{restaurant}', skipping.")
            continue

        try:
            merged_df = merged_df[merged_df['text'].notna()]
            merged_df = merged_df[merged_df['text'].str.strip() != '']

            # Deduplication
            existing_docs = list(output_collection.find({}, {"text": 1, "date": 1}))
            if existing_docs:
                existing_df = pd.DataFrame(existing_docs)
                existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce')
                merged_df = pd.merge(
                    merged_df,
                    existing_df,
                    on=["text", "date"],
                    how="left",
                    indicator=True
                )
                new_rows = merged_df[merged_df["_merge"] == "left_only"].drop(columns=["_merge"])
            else:
                new_rows = merged_df

            if new_rows.empty:
                save_log_to_buffer(f"🟡 No unique reviews for '{restaurant}' after deduplication.", "info")
                print(f"🟡 No unique reviews for '{restaurant}' after deduplication.")
                continue

            print(f"🧠 Running sentiment analysis for '{restaurant}'...")
            save_log_to_buffer(f"🧠 Running sentiment analysis for '{restaurant}'", "info")
            sentiment_df = run_sentiment_analysis_pipeline(new_rows, sample_size=None)
            print(f"✅ Sentiment complete. {len(sentiment_df)} rows")
            if 'date' in sentiment_df.columns:
                sentiment_df = sentiment_df.dropna(subset=['date'])

            # (Optional) Save to CSV - you can comment this if not needed
            path = os.path.join(output_db_name, f"{restaurant}2.csv")
            # sentiment_df.to_csv(path, index=False, encoding='utf-8-sig')

            if not sentiment_df.empty:
                output_collection.insert_many(sentiment_df.fillna("").to_dict(orient="records"))
                save_log_to_buffer(f"☁️ Inserted {len(sentiment_df)} to MongoDB: {output_collection_name}", "info")
                print(f"☁️ Inserted {len(sentiment_df)} to MongoDB: {output_collection_name}")

        except Exception as e:
            save_log_to_buffer(f"❌ Error processing '{restaurant}': {e}", "error")
            print(f"❌ Error processing '{restaurant}': {e}")
            save_logs_to_mongodb()

        end_time = time.time()
        print(f"⏱️ Done with '{restaurant}' in {round(end_time - start_time, 2)}s")

    save_log_to_buffer("🎉 All restaurants processed successfully.", "info")
    print("\n🎉 All restaurants processed successfully.")
    save_logs_to_mongodb()

# Entry point
if __name__ == "__main__":
    sentipulseRunThePipeline()
