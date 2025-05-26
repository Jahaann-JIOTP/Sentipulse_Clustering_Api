import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import optuna
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import List, Dict, Optional, Tuple, Any, Union
import concurrent.futures
import functools
import time
from tqdm import tqdm
import torch
import warnings

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:cisco123@13.234.241.103:27017/sentipulse?authSource=admin&readPreference=primary&ssl=false&directConnection=true")
DB_NAME = "sentipulse"  # Fixed DB name as specified

# Define constants
ASPECTS = ['food', 'service', 'price', 'ambiance', 'others']
SENTIMENTS = ['positive', 'negative', 'neutral']
PLATFORMS = ['google', 'foodpanda', 'uber_eats', 'zomato', 'all']  # Add more platforms as needed

print("📦 Loading SentenceTransformer model globally...")

try:
    GLOBAL_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
except Exception as e:
    print(f"❌ Global model load failed: {e}")
    GLOBAL_MODEL = None

# Add caching support
class LRUCache:
    """Simple LRU cache for embedding and cluster naming results"""
    
    def __init__(self, capacity=1000):
        self.cache = {}
        self.capacity = capacity
        self.usage_count = 0

    def get(self, key):
        if key in self.cache:
            self.cache[key]['count'] = self.usage_count
            self.usage_count += 1
            return self.cache[key]['value']
        return None

    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            # Remove least recently used item
            lru_key = min(self.cache.items(), key=lambda x: x[1]['count'])[0]
            del self.cache[lru_key]

        self.cache[key] = {'value': value, 'count': self.usage_count}
        self.usage_count += 1


class RestaurantReviewClustering:
    """
    A class to handle restaurant review clustering including data loading,
    text processing, embedding generation, clustering, and naming clusters.
    """

    def __init__(self, restaurant_name: str, random_seed: int = 42):
        """
        Initialize the clustering system.
        Args:
            restaurant_name: Name of the restaurant (used for collection naming)
            random_seed: Seed for random operations to ensure reproducibility
        """
        self.restaurant_name = restaurant_name.lower()  # Ensure lowercase
        self.collection_name = f"{self.restaurant_name}_sentimented_output"

        # Initialize model with proper error handling
        self.model = GLOBAL_MODEL
        if not self.model:
            raise RuntimeError("SentenceTransformer model not available.")


        self.mongo_client = None
        self.random_seed = random_seed

        # Set random seed for numpy to ensure reproducibility
        np.random.seed(self.random_seed)

        # Initialize caches
        self.embedding_cache = LRUCache(capacity=10000)
        self.cluster_name_cache = LRUCache(capacity=1000)

        # Create thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        try:
            self.mongo_client = MongoClient(MONGO_URI)
            # Test connection
            self.mongo_client.server_info()
            print("MongoDB connection successful")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            self.mongo_client = None

   
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def get_sample_reviews(self, texts: List[str], max_samples: int = 100) -> List[str]:
        if len(texts) <= max_samples:
            return texts
        sorted_texts = sorted(texts)
        indices = np.linspace(0, len(sorted_texts) - 1, max_samples, dtype=int)
        return [sorted_texts[i] for i in indices]

    def _batch_encode(self, texts: List[str], batch_size=64):
        """Encode texts in batches to improve efficiency"""
        if not self.model:
            raise RuntimeError("Model not initialized properly")

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                try:
                    batch_embeddings = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                except Exception as e:
                    print(f"❌ Error encoding batch {i // batch_size}: {e}")
                    batch_embeddings = np.zeros((len(batch), 384))

                all_embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch {i // batch_size}: {e}")
                # Create zero embeddings as fallback
                fallback_embeddings = np.zeros((len(batch), 384))  # 384 is typical for MiniLM-L6-v2
                all_embeddings.append(fallback_embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def encode_with_cache(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with caching for efficiency"""
        if not self.model:
            raise RuntimeError("Model not initialized properly")

        # Check cache first for each text
        embeddings = []
        texts_to_encode = []
        indices_to_fill = []

        for i, text in enumerate(texts):
            text_hash = hash(text)
            cached_embedding = self.embedding_cache.get(text_hash)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                texts_to_encode.append(text)
                indices_to_fill.append(i)

        # If all embeddings were cached, return them
        if not texts_to_encode:
            return np.array(embeddings)

        # Otherwise, encode the remaining texts
        new_embeddings = self._batch_encode(texts_to_encode)

        # Update cache with new embeddings
        for text, embedding in zip(texts_to_encode, new_embeddings):
            self.embedding_cache.put(hash(text), embedding)

        # Create the final embeddings array
        final_embeddings = np.zeros((len(texts), new_embeddings.shape[1]))

        # Fill in cached embeddings
        cached_idx = 0
        for i in range(len(texts)):
            if i in indices_to_fill:
                final_embeddings[i] = new_embeddings[indices_to_fill.index(i)]
            else:
                final_embeddings[i] = embeddings[cached_idx]
                cached_idx += 1

        return final_embeddings

    def name_cluster_with_llm(self, texts: List[str], aspect_name: str, sentiment: str = "positive") -> str:
        """Name a cluster using LLM with caching for repeated requests"""
        # Create a deterministic key for caching
        texts_sorted = sorted(texts)
        cache_key = f"{aspect_name}_{sentiment}_{hash(tuple(texts_sorted[:5]))}"

        # Check cache first
        cached_name = self.cluster_name_cache.get(cache_key)
        if cached_name is not None:
            return cached_name

        sample_reviews = self.get_sample_reviews(texts, max_samples=100)
        sample_str = "\n".join([f"- \"{review}\"" for review in sample_reviews])

        prompt = f"""
        You are helping a restaurant owner understand what customers are actually saying in a cluster of reviews.
        You're analyzing {sentiment} comments specifically about {aspect_name}.
        Here is a sample of what customers are saying:
        <<<{sample_str}>>>
        Your task:
        - Generate a 2–6 word cluster name that:
          1. Clearly summarizes what customers are talking about (e.g., portion size, seating comfort, ambience, wait time, etc.)
          2. Captures the actual experience or issue customers are facing (e.g., small portions, noisy atmosphere, long waits)
          3. Is clear, simple, and action-focused — like how you'd explain the issue to a friend or staff
          4. Helps the owner instantly understand the situation or issue — NO technical jargon
          5. If there are only 5 or fewer reviews, be extra careful — consider the entire context, not just one review
          6. DO NOT use general labels like "poor service" or "bad quality"
          7. DO NOT use overly specific food or cuisine terms like "Chinese, soup, noodles" etc
          8. DO NOT base the name on only one review — the name must reflect the entire cluster
          9. Your answer MUST be only 2–6 words — NO extra explanation, no intro, no summary
        JUST reply with the 2–6 word CLUSTER NAME. NOTHING ELSE.
        """

        if not GROQ_API_KEY:
            print("Warning: GROQ_API_KEY not found. Using fallback cluster naming.")
            result = f"{aspect_name.capitalize()} {sentiment.capitalize()} Cluster {np.random.randint(1, 100)}"
            self.cluster_name_cache.put(cache_key, result)
            return result

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You create short, action-focused labels..."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "top_p": 1.0
        }

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data
            )
            resp.raise_for_status()
            cluster_name = resp.json()['choices'][0]['message']['content'].strip()
            cluster_name = re.sub(r'^["\']+|["\']+$|[.]$', '', cluster_name).strip()

            if " or " in cluster_name:
                cluster_name = cluster_name.split(" or ")[0].strip()

            words = cluster_name.split()
            if len(words) > 6:
                cluster_name = " ".join(words[:6])

            # Store in cache
            self.cluster_name_cache.put(cache_key, cluster_name)
            return cluster_name

        except Exception as e:
            print(f"Cluster naming failed: {e}")
            fallback = f"{aspect_name.capitalize()} {sentiment.capitalize()} Cluster"
            self.cluster_name_cache.put(cache_key, fallback)
            return fallback

    def load_data(
            self,
            aspect: str,
            sentiment: str,
            start_date: str = None,
            end_date: str = None,
            restaurant_id: str = None,
            platform: str = "all"  # New parameter for platform filtering
    ) -> pd.DataFrame:
        """
        Load data from MongoDB with optional platform filtering

        Args:
            aspect: The aspect to analyze
            sentiment: The sentiment to filter by
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            restaurant_id: Restaurant ID for filtering
            platform: Platform to filter by ('google', 'foodpanda', 'uber_eats', 'zomato', 'all')
        """
        if not self.mongo_client:
            raise ConnectionError("MongoDB connection not available")

        db = self.mongo_client[DB_NAME]
        col = db[self.collection_name]

        query: Dict[str, Any] = {}

        # Date filtering
        if start_date or end_date:
            dq: Dict[str, Any] = {}
            if start_date:
                try:
                    dq["$gte"] = datetime.strptime(start_date, "%Y-%m-%d")
                except ValueError:
                    print(f"Invalid start_date format: {start_date}")
                    return pd.DataFrame()

            if end_date:
                try:
                    dq["$lte"] = datetime.strptime(end_date, "%Y-%m-%d")
                except ValueError:
                    print(f"Invalid end_date format: {end_date}")
                    return pd.DataFrame()

            if dq:
                query["date"] = dq

        # Restaurant ID filtering
        if restaurant_id:
            query["restaurant_id"] = restaurant_id

        # Platform filtering - NEW FUNCTIONALITY
        if platform and platform.lower() != "all":
            query["sent_platform"] = platform.lower()
            print(f"Filtering by platform: {platform}")

        try:
            # Add projection to only fetch needed fields for better performance
            projection = {
                "sent_sentiment": 1,
                "sent_platform": 1,  # Include platform in projection
                "date": 1,
                "restaurant_id": 1
            }

            # Add aspect-specific fields with sent_ prefix
            for asp in ASPECTS:
                projection[f"sent_{asp}_part"] = 1
                projection[f"sent_{asp}_sentiment"] = 1

            docs = list(col.find(query, projection))
        except Exception as e:
            print(f"Error querying MongoDB: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(docs) if docs else pd.DataFrame()
        print(f"Loaded {len(df)} reviews" + (f" from platform: {platform}" if platform != "all" else ""))

        # Filter by overall sentiment (updated column name)
        if sentiment and "sent_sentiment" in df.columns:
            df = df[df["sent_sentiment"] == sentiment]
            print(f"Overall sentiment filter: {len(df)} left")

        # Filter by aspect-specific sentiment (updated column name)
        asp_col = f"sent_{aspect}_sentiment"
        if sentiment and asp_col in df.columns:
            df = df[df[asp_col] == sentiment]
            print(f"{aspect} sentiment filter: {len(df)} left")

        return df

    def optimize_clustering_params(self, embeddings, max_n_clusters, n_trials=10):
        """Optimize clustering parameters using Optuna"""

        def objective(trial):
            max_c = min(max_n_clusters, len(embeddings) - 1)
            if max_c < 2:
                return -1

            n = trial.suggest_int("n_clusters", 2, max_c)
            km = KMeans(n_clusters=n, n_init=10, random_state=self.random_seed)

            try:
                km.fit(embeddings)
                return silhouette_score(embeddings, km.labels_)
            except:
                return -1

        # Create a faster sampler
        sampler = optuna.samplers.TPESampler(seed=self.random_seed, n_startup_trials=min(3, n_trials))
        study = optuna.create_study(direction='maximize', sampler=sampler)

        # Limit the number of trials for faster execution
        actual_trials = min(n_trials, 5)
        study.optimize(objective, n_trials=actual_trials)

        return study.best_params.get('n_clusters', 2)

    def cluster_reviews(
            self,
            df: pd.DataFrame,
            aspect: str,
            sentiment: str,
            n_trials: int = 5,  # Reduced from 10
            visualize: bool = False
    ) -> pd.DataFrame:
        """Cluster reviews with optimized parallel processing"""
        part_col = f"sent_{aspect}_part"  # Updated with sent_ prefix

        if part_col not in df.columns:
            print(f"Column {part_col} not found")
            return df

        df = df.dropna(subset=[part_col])

        if len(df) < 5:
            print(f"Not enough data ({len(df)} rows)")
            return df

        df = df.sort_values(by=part_col).reset_index(drop=True)
        texts = df[part_col].astype(str).tolist()

        # Generate embeddings with caching
        start_time = time.time()
        try:
            embeddings = self.encode_with_cache(texts)
            print(f"Embedding generation took {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return df

        # Optimize clustering parameters with fewer trials for speed
        start_time = time.time()
        max_n_clusters = min(5, len(texts) - 1)
        n_clusters = self.optimize_clustering_params(embeddings, max_n_clusters, n_trials)
        print(f"Parameter optimization took {time.time() - start_time:.2f} seconds")

        # Perform the final clustering
        start_time = time.time()
        km_final = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_seed)
        km_final.fit(embeddings)
        labels = km_final.labels_
        print(f"Final clustering took {time.time() - start_time:.2f} seconds")

        # Group texts by cluster
        cluster_texts = {i: [] for i in range(n_clusters)}
        for idx, lbl in enumerate(labels):
            cluster_texts[lbl].append(texts[idx])

        # Name clusters in parallel
        start_time = time.time()
        print(f"Naming {n_clusters} clusters...")

        # Use parallel execution for cluster naming
        naming_futures = {}
        for i in range(n_clusters):
            naming_futures[i] = self.executor.submit(
                self.name_cluster_with_llm, cluster_texts[i], aspect, sentiment
            )

        # Collect results
        cluster_names = {}
        for i, future in naming_futures.items():
            try:
                cluster_names[i] = future.result()
                print(f"Cluster {i}: {cluster_names[i]}")
            except Exception as e:
                print(f"Error naming cluster {i}: {e}")
                cluster_names[i] = f"{aspect.capitalize()} {sentiment.capitalize()} Cluster {i}"

        print(f"Cluster naming took {time.time() - start_time:.2f} seconds")

        # Calculate distribution
        counts = pd.Series(labels).value_counts(normalize=True) * 100

        # Add results to dataframe
        df[f"{aspect}_{sentiment}_cluster"] = labels
        df[f"{aspect}_{sentiment}_cluster_name"] = df[f"{aspect}_{sentiment}_cluster"].map(cluster_names)
        df[f"{aspect}_{sentiment}_distribution"] = df[f"{aspect}_{sentiment}_cluster"].map(
            lambda x: f"{counts.get(x, 0):.1f}%")

        return df

    def process_aspect(
            self,
            aspect: str,
            sentiment: str,
            df: pd.DataFrame,
            visualize: bool = False
    ) -> pd.DataFrame:
        """Process a single aspect from pre-loaded data"""
        print(f"\n{'=' * 50}")
        print(f"Processing aspect: {aspect} - {sentiment}")
        print(f"{'=' * 50}")

        # For each aspect, we don't need to reload the data
        # but we do need to ensure we have the right filtering
        aspect_df = df.copy()

        # Apply aspect-specific sentiment filtering (updated column name)
        asp_col = f"sent_{aspect}_sentiment"
        if sentiment and asp_col in aspect_df.columns:
            aspect_df = aspect_df[aspect_df[asp_col] == sentiment]
            print(f"{aspect} sentiment filter: {len(aspect_df)} rows")

        if aspect_df.empty:
            print(f"No data for {aspect} after filtering")
            return pd.DataFrame()

        # Run clustering for this aspect
        clustered_df = self.cluster_reviews(aspect_df, aspect, sentiment, visualize=visualize)

        # These are the expected cluster columns
        cluster_cols = [
            f"{aspect}_{sentiment}_cluster",
            f"{aspect}_{sentiment}_cluster_name",
            f"{aspect}_{sentiment}_distribution"
        ]

        # If no clustering occurred, skip this aspect
        if not all(col in clustered_df.columns for col in cluster_cols):
            print(f"Skipping aspect '{aspect}' due to insufficient data or missing cluster columns.")
            return pd.DataFrame()

        # Extract just the cluster columns for this aspect
        if '_id' in clustered_df.columns:
            cluster_results = clustered_df[['_id'] + cluster_cols].copy()
        else:
            cluster_results = clustered_df.reset_index()[['index'] + cluster_cols].copy()

        # Print summary
        if f"{aspect}_{sentiment}_cluster_name" in clustered_df.columns:
            print(f"\nCluster Distribution for {aspect}:")
            name_counts = clustered_df[f"{aspect}_{sentiment}_cluster_name"].value_counts()
            for name, count in name_counts.items():
                if pd.notna(name):  # Only show non-null cluster names
                    print(f"- {name}: {count} reviews")

        return cluster_results

    def run_clustering(
            self,
            aspect: str,
            sentiment: str,
            start_date: str = None,
            end_date: str = None,
            restaurant_id: str = None,
            platform: str = "all",  # New parameter
            visualize: bool = False,
            save_output: bool = True,
            hash_inputs: bool = True
    ) -> pd.DataFrame:
        """
        Run clustering for a single aspect with optional platform filtering

        Args:
            aspect: Aspect to analyze
            sentiment: Sentiment to filter by
            start_date: Start date for filtering
            end_date: End date for filtering
            restaurant_id: Restaurant ID for filtering
            platform: Platform to filter by ('google', 'foodpanda', 'uber_eats', 'zomato', 'all')
            visualize: Whether to create visualizations
            save_output: Whether to save output to CSV
            hash_inputs: Whether to hash inputs for reproducibility
        """
        if aspect not in ASPECTS:
            raise ValueError(f"Aspect must be one of {ASPECTS}")
        if sentiment not in SENTIMENTS:
            raise ValueError(f"Sentiment must be one of {SENTIMENTS}")

        if hash_inputs:
            import hashlib
            inp = f"{start_date}{end_date}{restaurant_id}{sentiment}{aspect}{platform}"
            h = hashlib.md5(inp.encode()).hexdigest()[:8]
            print(f"Input hash: {h}")
            seed_val = int(h, 16) % (2 ** 32 - 1)
            np.random.seed(seed_val)
            self.random_seed = seed_val

        df = self.load_data(aspect, sentiment, start_date, end_date, restaurant_id, platform)

        if df.empty:
            print("No data after filtering")
            return df

        print(f"Clustering {aspect} - {sentiment}" + (f" (Platform: {platform})" if platform != "all" else ""))
        result_df = self.cluster_reviews(df, aspect, sentiment, visualize=visualize)

        if save_output and not result_df.empty:
            os.makedirs("dataWithClusters", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            platform_suffix = f"_{platform}" if platform != "all" else ""
            out = f"dataWithClusters/{self.restaurant_name}_{aspect}_{sentiment}{platform_suffix}_{ts}_clustered_output.csv"
            result_df.to_csv(out, index=False)
            print(f"Saved to {out}")

        return result_df

    def run_multi_aspect_clustering(
            self,
            aspects: List[str],
            sentiment: str,
            start_date: str = None,
            end_date: str = None,
            restaurant_id: str = None,
            platform: str = "all",  # New parameter
            visualize: bool = False,
            save_output: bool = True,
            hash_inputs: bool = True,
            parallel: bool = True  # New parameter to enable/disable parallel processing
    ) -> pd.DataFrame:
        """
        Run clustering for multiple aspects with the specified sentiment and optional platform filtering.

        Args:
            aspects: List of aspects to analyze
            sentiment: Sentiment to analyze
            start_date: Start date for filtering reviews
            end_date: End date for filtering reviews
            restaurant_id: Restaurant ID for filtering reviews
            platform: Platform to filter by ('google', 'foodpanda', 'uber_eats', 'zomato', 'all')
            visualize: Whether to create visualizations
            save_output: Whether to save the output to CSV
            hash_inputs: Whether to hash inputs for reproducibility
            parallel: Whether to process aspects in parallel

        Returns:
            DataFrame with clustering results for all aspects
        """
        start_time_total = time.time()

        # Validate aspects
        valid_aspects = [aspect for aspect in aspects if aspect in ASPECTS]
        if not valid_aspects:
            raise ValueError(f"No valid aspects provided. Must be one or more of {ASPECTS}")

        if sentiment not in SENTIMENTS:
            raise ValueError(f"Sentiment must be one of {SENTIMENTS}")

        # Load data once - we'll reuse the same dataset for all aspects
        print("Loading data from MongoDB...")
        initial_df = self.load_data(valid_aspects[0], sentiment, start_date, end_date, restaurant_id, platform)
        if initial_df.empty:
            print("No data after filtering")
            return initial_df

        result_df = initial_df.copy()

        if parallel and len(valid_aspects) > 1:
            # Process aspects in parallel
            print(f"Processing {len(valid_aspects)} aspects in parallel...")
            futures = {}

            for aspect in valid_aspects:
                if hash_inputs:
                    import hashlib
                    inp = f"{start_date}{end_date}{restaurant_id}{sentiment}{aspect}{platform}"
                    h = hashlib.md5(inp.encode()).hexdigest()[:8]
                    print(f"Input hash for {aspect}: {h}")
                    # Note: We're not changing the global random seed here to avoid race conditions

                # Submit task to thread pool
                futures[aspect] = self.executor.submit(
                    self.process_aspect, aspect, sentiment, result_df.copy(), visualize
                )

            # Collect results
            aspect_results = {}
            for aspect, future in futures.items():
                try:
                    aspect_results[aspect] = future.result()
                except Exception as e:
                    print(f"Error processing aspect {aspect}: {e}")
                    aspect_results[aspect] = pd.DataFrame()

            # Merge all aspect results into the final DataFrame
            for aspect, aspect_df in aspect_results.items():
                if not aspect_df.empty and f"{aspect}_{sentiment}_cluster" in aspect_df.columns:
                    # Merge with the result DataFrame
                    if '_id' in result_df.columns and '_id' in aspect_df.columns:
                        result_df = pd.merge(result_df, aspect_df, on='_id', how='left')
                    else:
                        # If no _id column, merge on index
                        result_df = pd.merge(result_df, aspect_df, left_index=True, right_on='index', how='left')
                        if 'index' in result_df.columns:
                            result_df = result_df.drop('index', axis=1)

        # Save the final output with all aspects' clustering results
        if save_output and not result_df.empty:
            os.makedirs("dataWithClusters", exist_ok=True)
            aspect_str = "_".join(valid_aspects)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            platform_suffix = f"_{platform}" if platform != "all" else ""
            out = f"dataWithClusters/{self.restaurant_name}_multi_{aspect_str}_{sentiment}{platform_suffix}_{ts}_clustered_output.csv"
            result_df.to_csv(out, index=False)
            print(f"\nSaved multi-aspect clustering results to {out}")

        print(f"Total processing time: {time.time() - start_time_total:.2f} seconds")
        return result_df


def main():
    restaurant_name = "ginyaki"
    script_start = time.time()
    clustering = RestaurantReviewClustering(restaurant_name=restaurant_name)
    print(f"Initialization took {time.time() - script_start:.2f} seconds")

    # Define list of aspects to analyze
    aspects_to_analyze = ["food", "service", "price", "ambiance",
                          "others"]  # Can be modified to include any subset of ASPECTS
    sentiment_to_analyze = "positive"

    # NEW: Platform filtering options
    # Options: "google", "foodpanda", "uber_eats", "zomato", "all"
    platform_to_analyze = "google"  # Change this to filter by specific platform

    print(f"Analyzing platform: {platform_to_analyze}")

    # Run multi-aspect clustering with parallel processing and platform filtering
    df_result = clustering.run_multi_aspect_clustering(
        aspects=aspects_to_analyze,
        sentiment=sentiment_to_analyze,
        start_date="2025-05-19",
        end_date="2025-05-26",
        platform=platform_to_analyze,  # NEW: Platform filtering
        save_output=False,
        hash_inputs=False,
        parallel=True  # Enable parallel processing
    )

    print(f"\nCompleted clustering for aspects: {aspects_to_analyze} ({sentiment_to_analyze})")
    if platform_to_analyze != "all":
        print(f"Platform filtered: {platform_to_analyze}")
    print(f"Processed {len(df_result)} reviews in total")

    # Summary of all clusters across all aspects
    print("\nSummary of Clusters Across All Aspects:")
    for aspect in aspects_to_analyze:
        col = f"{aspect}_{sentiment_to_analyze}_cluster_name"
        if col in df_result.columns:
            print(f"\n{aspect.upper()} CLUSTERS:")
            dist = df_result[col].value_counts()
            for name, cnt in dist.items():
                if pd.notna(name):  # Only show non-null cluster names
                    print(f"- {name}: {cnt} reviews")

    # Show platform distribution if available
    if 'sent_platform' in df_result.columns:
        print(f"\nPlatform Distribution:")
        platform_dist = df_result['sent_platform'].value_counts()
        for platform, count in platform_dist.items():
            if pd.notna(platform):
                print(f"- {platform}: {count} reviews")

    return df_result


if __name__ == "__main__":
    # Run the main clustering analysis
    result_df = main()

