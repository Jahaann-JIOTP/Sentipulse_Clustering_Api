import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection setup
uri = os.getenv('MONGO_URI')
client = MongoClient(uri)
db = client["sentipulse"]

# A variable to accumulate logs

log_buffer = []
"""updated_df = pd.concat([pd.DataFrame(existing_docs), sentiment_df], ignore_index=True)
            # updated_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            # print(f"✅ CSV saved: {output_path}")"""
def save_log_to_buffer(log_message: str, log_type: str):
    """Add log messages to an in-memory buffer."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_entry = {
            "timestamp": timestamp,
            "log_type": log_type,  # e.g., 'error', 'info', etc.
            "message": log_message
        }
        log_buffer.append(log_entry)  # Store log in buffer
    except Exception as e:
        print(f"❌ Error adding log to buffer: {e}")

def save_logs_to_mongodb():
    """Save all accumulated logs to MongoDB in bulk."""
    try:
        if log_buffer:
            log_collection = db["sentipulse_script_logs"]  # or any specific collection for logs
            result = log_collection.insert_many(log_buffer)
            print(f"☁️ {len(result.inserted_ids)} logs saved to MongoDB.")
            # Clear the buffer after saving to MongoDB
            log_buffer.clear()
        else:
            print("⚠️ No logs to save.")
    except Exception as e:
        print(f"❌ Error saving logs to MongoDB: {e}")

def standardize_df(df: pd.DataFrame, platform:str) -> pd.DataFrame:
    """Standardizes '_id', 'text', and 'date' columns for consistency."""
    try:
        # Ensure '_id' column is dropped if it exists
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
            df = standardize_price_bins(df, platform)

        # Ensure 'text' column exists and is of string type
        if 'text' not in df.columns:
            df['text'] = ""
        df['text'] = df['text'].astype(str)

        # Ensure 'date' column exists and is converted to datetime
        if 'date' not in df.columns:
            df['date'] = pd.NaT
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # invalid strings become NaT
        df['date'] = df['date'].apply(
            lambda x: x.tz_localize(None) if pd.notnull(x) and hasattr(x, 'tz_localize') and x.tzinfo else x)
        df = df.dropna(subset=['date'])
        df = df.sort_values(by='date', ascending=True)  # Ensure correct sorting

        # Ensure restaurant and platform columns exist for consistency (even if not in original data)
        if 'restaurant' not in df.columns:
            df['restaurant'] = ""
        if 'platform' not in df.columns:
            df['platform'] = ""

        return df

    except Exception as e:
        # Log error to buffer instead of directly to MongoDB
        save_log_to_buffer(f"❌ Error during standardization of dataframe: {e}", "error")
        raise  # Re-raise the exception for higher-level handling



#-----------------------------------------------------------------------------------------------------------------------
def standardize_price_bins(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    If collection is 'foodpanda', converts price_per_person numeric values
    into standard labeled bins like 'Rs 1–500', 'Rs 501–1000', ..., 'Rs 5000+'.
    Ignores NaNs and invalids.
    """

    if platform != "foodpanda":
        return df  # No processing needed for other collections

    def bin_price(value):
        try:
            if pd.isna(value):
                return np.nan
            value = float(value)
            if value <= 500:
                return "Rs 0–500"
            elif value <= 1000:
                return "Rs 500–1000"
            elif value <= 1500:
                return "Rs 1000–1500"
            elif value <= 2000:
                return "Rs 1500–2000"
            elif value <= 2500:
                return "Rs 2000–2500"
            elif value <= 3000:
                return "Rs 2500–3000"
            elif value <= 3500:
                return "Rs 3000–3500"
            elif value <= 4000:
                return "Rs 3500–4000"
            elif value <= 5000:
                return "Rs 4000–5000"
            else:
                return "Rs 5000+"
        except Exception:
            return np.nan  # If value is not convertible to float

    if 'price_per_person' in df.columns:
        df['price_per_person'] = df['price_per_person'].apply(bin_price)

    return df


