import os
import re
import math
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from modules.utils import save_log_to_buffer, save_logs_to_mongodb

# === Load environment ===
load_dotenv()
groq_key = os.getenv('GROQ_API_KEY')

# === LLM Setup ===
if not groq_key:
    raise ValueError("GROQ_API_KEY not set in environment variables.")

llm = ChatGroq(
    model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    temperature=0.2,
    max_tokens=8192
)

# === Prompt Template ===
BATCH_TEMPLATE = """
  You are a professional language model trained for detailed aspect-based sentiment analysis of restaurant customer reviews.

  Your ONLY job is to return a clean, valid **JSON array of objects**—each representing one review—following the structure and rules described below. 

  🔒 **STRICT RULES (Do NOT break these):**
  - ❌ Do NOT add explanations, apologies, or any other text outside the JSON.
  - ✅ ONLY return the JSON output. No comments, no summaries, no "Here's your response", etc.
  - ❌ If a review is unclear or unprocessable, return the default empty object as specified below. DO NOT generate fake or uncertain interpretations.
  - 🛑 If unsure, return a conservative neutral sentiment with empty aspect details or fallback to the "empty review" structure.
  - ⚠️ IMPORTANT: Return exactly the same number of JSON objects as there are reviews in the input.

  ---

  🎯 **Analysis Goals per Review**:

  1. **Overall Sentiment**: One of:
     - "positive", "negative", or "neutral"

  2. **Sentiment Phrase**: A snippet that reflects why the sentiment was chosen

  3. **Aspects** (mention detection):
     - "food" (covers quality, taste, quantity, dishes, ingredients)
     - "ambiance" (covers environment, atmosphere, cleanliness, decor)
     - "price" (covers cost, value for money, affordability)
     - "service" (covers staff, waiters, speed, attentiveness)
     - Use "others" **only** if NONE of the above aspects apply

  4. **Aspect Details** (for each identified aspect):
     - **part**: Text supporting that aspect
     - **sentiment**: "positive", "negative", or "neutral"
     - **rating**: Float from 1 to 5

  5. **Dominant Words** (from the entire review):
     - **positive_words**: List of 3-7 dominant positive words/phrases from the review (if any)
     - **negative_words**: List of 3-7 dominant negative words/phrases from the review (if any)
     - **neutral_words**: List of 3-7 dominant neutral words/phrases from the review (if any)

  6. **Unrelated or unclear text**:
     - ⚠️ Even if the text is NOT a review (e.g., song lyrics, nonsense, ads), still return a structured output.
     - Use neutral sentiment, "others" as the aspect, and 3 as the default rating.
     - Do NOT leave anything empty—ALWAYS return a structured response for each input review.

  7. **Empty Review Handling**: If the review is blank, return:

  {{
    "overall_sentiment": "",
    "part": "",
    "overall_aspects": [],
    "overall_rating": 0,
    "aspects": {{}},
    "positive_words": [],
    "negative_words": [],
    "neutral_words": []
  }}

  8. **Overall Rating**: Float (1 decimal) reflecting the total impression (e.g., 4)

  ---

  📦 **Required JSON Format (Per Review):**

  {{
    "overall_sentiment": "positive/negative/neutral",
    "part": "text reflecting sentiment",
    "overall_aspects": ["aspect1", "aspect2"],
    "overall_rating": 4,
    "aspects": {{
      "aspect_name": {{
        "part": "snippet related to aspect",
        "sentiment": "positive/negative/neutral",
        "rating": 4
      }}
    }},
    "positive_words": ["delicious", "fantastic", "enjoyable"],
    "negative_words": ["overpriced", "slow"],
    "neutral_words": ["restaurant", "menu", "setting"]
  }}

  ---

  📌 **Important**:
  - ALWAYS return exactly the same number of review objects as provided in the input.
  - Never explain yourself. Output JSON and nothing else.
  - For dominant words, extract only the most significant words or short phrases from the review.
  - Words should be categorized based on their sentiment in the context of the review.
  - for the words one more thing if u r taking the positive negative or neutral 
  make sure that the words are according to restaurant context not other then that like "above par"

  ---

  🔍 **Examples**:

  Review: "The pasta was absolutely delicious but the prices were too high for what you get."
  {{
    "overall_sentiment": "positive",
    "part": "The pasta was absolutely delicious but the prices were too high for what you get.",
    "overall_aspects": ["food", "price"],
    "overall_rating": 3,
    "aspects": {{
      "food": {{
        "part": "The pasta was absolutely delicious",
        "sentiment": "positive",
        "rating": 4
      }},
      "price": {{
        "part": "prices were too high for what you get",
        "sentiment": "negative",
        "rating": 2
      }}
    }},
    "positive_words": ["delicious", "pasta"],
    "negative_words": ["too high", "prices"],
    "neutral_words": ["what you get"]
  }}

  ---

  📥 Now analyze the following reviews:

  {reviews_block}
  """
# Use your full prompt content here
prompt = PromptTemplate.from_template(BATCH_TEMPLATE)
chain = ({"reviews_block": RunnablePassthrough()} | prompt | llm | StrOutputParser())


# === Analysis Helpers ===
def prepare_reviews_batch(review_texts):
    return [text.strip() if text and text.strip() else "" for text in review_texts]


def parse_llm_output(output, expected_count):
    start = output.find('[')
    end = output.rfind(']')
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in LLM output.")
    cleaned = output[start:end + 1]
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("Parsed output is not a list.")

        # Check if we got the expected number of results
        if len(parsed) != expected_count:
            save_log_to_buffer(f"⚠️ Warning: Expected {expected_count} results but got {len(parsed)}", "warning")
            # If we got too many results, truncate
            if len(parsed) > expected_count:
                parsed = parsed[:expected_count]
            # If we got too few results, pad with empty objects
            elif len(parsed) < expected_count:
                parsed.extend([{
                    "overall_sentiment": "",
                    "part": "",
                    "overall_aspects": [],
                    "overall_rating": 0,
                    "aspects": {},
                    "positive_words": [],
                    "negative_words": [],
                    "neutral_words": []
                } for _ in range(expected_count - len(parsed))])

        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e}")


def analyze_reviews_batch(review_texts):
    review_texts = prepare_reviews_batch(review_texts)
    expected_count = len(review_texts)

    if all(text == "" for text in review_texts):
        return [{
            "overall_sentiment": "",
            "part": "",
            "overall_aspects": [],
            "overall_rating": 0,
            "aspects": {},
            "positive_words": [],
            "negative_words": [],
            "neutral_words": []
        } for _ in review_texts]

    try:
        reviews_str = "\n".join([f"Review {i + 1}: {text}" for i, text in enumerate(review_texts)])
        output = chain.invoke(reviews_str)
        parsed_results = parse_llm_output(output, expected_count)

        for result in parsed_results:
            result['overall_sentiment'] = result.get('overall_sentiment', '').lower()
            if result['overall_sentiment'] not in ['positive', 'negative', 'neutral']:
                result['overall_sentiment'] = ''
            result['overall_rating'] = int(round(result.get('overall_rating', 0)))
            aspects = result.get('aspects', {}) or {}
            result['aspects'] = aspects

            # Ensure word lists exist with defaults
            result['positive_words'] = result.get('positive_words', []) or []
            result['negative_words'] = result.get('negative_words', []) or []
            result['neutral_words'] = result.get('neutral_words', []) or []

            for aspect, aspect_data in aspects.items():
                aspect_data['sentiment'] = aspect_data.get('sentiment', 'neutral').lower()
                if aspect_data['sentiment'] not in ['positive', 'negative', 'neutral']:
                    aspect_data['sentiment'] = 'neutral'
                aspect_data['rating'] = int(round(aspect_data.get('rating', 0)))

        return parsed_results

    except Exception as e:
        save_log_to_buffer(f"❌ Failed to parse LLM output: {e}", "error")
        save_log_to_buffer(f"Raw LLM output (truncated): {output[:1000]}", "debug")
        return [{
            "overall_sentiment": "",
            "part": "",
            "overall_aspects": [],
            "overall_rating": 0,
            "aspects": {},
            "positive_words": [],
            "negative_words": [],
            "neutral_words": []
        } for _ in review_texts]  # Return empty objects for all inputs on error


# === Main Runner ===
def run_sentiment_analysis_pipeline(df, sample_size=None, batch_size=10):
    if sample_size:
        df = df.sample(sample_size).reset_index(drop=True)
        print(f"🔍 Sampled {sample_size} reviews")
        save_log_to_buffer(f"🔍 Sampled {sample_size} reviews", "info")
    else:
        print(f"🧠 Processing {len(df)} reviews")
        save_log_to_buffer(f"🧠 Processing {len(df)} reviews", "info")

    # Reset all columns that will store analysis results
    df['sent_sentiment'] = ""
    df['sent_part'] = ""
    df['sent_aspects'] = None
    df['sent_ratings'] = 0

    # Add new columns for dominant words
    df['sent_positive_words'] = None
    df['sent_negative_words'] = None
    df['sent_neutral_words'] = None

    # Updated consolidated aspect categories
    aspect_categories = ['food', 'ambiance', 'price', 'service', 'others']
    for cat in aspect_categories:
        df[f'sent_{cat}'] = 0
        df[f'sent_{cat}_sentiment'] = ""
        df[f'sent_{cat}_rating'] = 0
        df[f'sent_{cat}_part'] = ""
        # Removed the topic column

    num_reviews = len(df)
    for batch_idx in range(math.ceil(num_reviews / batch_size)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_reviews)
        batch_size_current = end_idx - start_idx

        print(f"🚀 Processing batch {batch_idx + 1}/{math.ceil(num_reviews / batch_size)}...")
        try:
            batch_reviews = df['text'].iloc[start_idx:end_idx].tolist()
            results = analyze_reviews_batch(batch_reviews)

            # Ensure we got the correct number of results for this batch
            if len(results) != batch_size_current:
                save_log_to_buffer(f"⚠️ Results count mismatch: expected {batch_size_current}, got {len(results)}",
                                   "warning")
                # Adjust the results to match the expected count
                if len(results) > batch_size_current:
                    results = results[:batch_size_current]
                else:
                    results.extend([{
                        "overall_sentiment": "",
                        "part": "",
                        "overall_aspects": [],
                        "overall_rating": 0,
                        "aspects": {},
                        "positive_words": [],
                        "negative_words": [],
                        "neutral_words": []
                    } for _ in range(batch_size_current - len(results))])

            for i, result in enumerate(results):
                idx = start_idx + i
                if idx >= len(df):
                    save_log_to_buffer(f"⚠️ Index {idx} out of bounds for DataFrame of length {len(df)}", "error")
                    continue

                df.at[idx, 'sent_sentiment'] = result.get('overall_sentiment', 'neutral')
                df.at[idx, 'sent_part'] = result.get('part', '')
                df.at[idx, 'sent_aspects'] = str(result.get('overall_aspects', []))
                df.at[idx, 'sent_ratings'] = result.get('overall_rating', 0)

                # Store dominant words
                df.at[idx, 'sent_positive_words'] = str(result.get('positive_words', []))
                df.at[idx, 'sent_negative_words'] = str(result.get('negative_words', []))
                df.at[idx, 'sent_neutral_words'] = str(result.get('neutral_words', []))

                for cat in aspect_categories:
                    df.at[idx, cat] = 1 if cat in result.get('overall_aspects', []) else 0
                    if cat in result.get('aspects', {}):
                        aspect_data = result['aspects'][cat]
                        df.at[idx, f'sent_{cat}_sentiment'] = aspect_data.get('sentiment', 'neutral')
                        df.at[idx, f'sent_{cat}_rating'] = aspect_data.get('rating', 0)
                        df.at[idx, f'sent_{cat}_part'] = aspect_data.get('part', '')
                        # Removed the topic column assignment

            print(f"✅ Batch {batch_idx + 1} processed with {len(results)} results for {batch_size_current} reviews.")
            save_log_to_buffer(
                f"✅ Batch {batch_idx + 1} processed with {len(results)} results for {batch_size_current} reviews",
                "info")

        except Exception as e:
            print(f"❌ Error in batch {batch_idx + 1}: {str(e)}")
            save_log_to_buffer(f"❌ Error in batch {batch_idx + 1}: {str(e)}", "error")

    print(f"🎉 Sentiment analysis completed for {len(df)} reviews.")
    save_log_to_buffer("🎉 Sentiment analysis complete", "info")
    save_logs_to_mongodb()
    return df