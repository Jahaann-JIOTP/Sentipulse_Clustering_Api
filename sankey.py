from flask import Flask, request, jsonify
from flask_cors import CORS
from review_clustering import RestaurantReviewClustering
from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route("/api/sankey/full", methods=["POST"])
def get_full_sankey_data():
    try:
        import pandas as pd  # Ensure this import exists

        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        restaurant = data.get("restaurant", "ginyaki")
        sentiment = data.get("sentiment", "negative")
        platform = data.get("platform", "all")
        

        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")

        duration_key = data.get("duration", "yesterday").lower()

        if duration_key == "yesterday":
            start_date = today - timedelta(days=1)
            end_date = start_date
        elif duration_key == "last week":
            start_date = today - timedelta(days=7)
            end_date = today
        elif duration_key == "this month":
            start_date = today.replace(day=1)
            end_date = today
        elif duration_key == "last month":
            first_day_this_month = today.replace(day=1)
            last_month_end = first_day_this_month - timedelta(days=1)
            start_date = last_month_end.replace(day=1)
            end_date = last_month_end
        else:
            return jsonify({"error": f"Unsupported duration: {duration_key}"}), 400


        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        clustering = RestaurantReviewClustering(restaurant_name=restaurant)

        df = clustering.run_multi_aspect_clustering(
            aspects=["food", "service", "price", "ambiance", "others"],
            sentiment=sentiment,
            start_date=start_date_str,
            end_date=end_date_str,
            platform=platform,
            save_output=False,
            hash_inputs=False,
            parallel=True
        )

        if df.empty:
            return jsonify([])

        # Mapping aspects to display names and colors
        aspect_mappings = {
            "food": {"label": "Food", "color": "#D82E5E"},
            "service": {"label": "Staff", "color": "#515CD6"},
            "price": {"label": "Cost", "color": "#E8B501"},
            "ambiance": {"label": "Ambiance", "color": "#D3642C"},
            "others": {"label": "Others", "color": "#8BC63E"},
        }

        sankey_links = []
        root_node = sentiment.capitalize()

        for aspect_key, config in aspect_mappings.items():
            cluster_name_col = f"{aspect_key}_{sentiment}_cluster_name"
            if cluster_name_col in df.columns:
                aspect_df = df[[f"sent_{aspect_key}_sentiment", cluster_name_col]]
                aspect_df = aspect_df[aspect_df[f"sent_{aspect_key}_sentiment"] == sentiment]

                if aspect_df.empty:
                    continue

                aspect_label = config["label"]
                color = config["color"]

                # Level 1: Sentiment → Aspect
                sankey_links.append({
                    "from": root_node,
                    "to": aspect_label,
                    "value": len(aspect_df),
                    "color": color
                })

                # Level 2: Aspect → Cluster Name
                cluster_counts = aspect_df[cluster_name_col].value_counts()
                for cluster_name, count in cluster_counts.items():
                    if pd.notna(cluster_name):
                        sankey_links.append({
                            "from": aspect_label,
                            "to": cluster_name,
                            "value": int(count),
                            "color": color
                        })

        return jsonify(sankey_links)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sankey", methods=["POST"])
def get_sankey_data():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        restaurant = data.get("restaurant", "ginyaki")
        sentiment = data.get("sentiment", "negative")
        platform = data.get("platform", "all")

        # Handle duration to calculate start and end dates
        duration = data.get("duration", {"durationType": "days", "durationValue": 7})
        duration_type = duration.get("durationType", "days")
        duration_value = int(duration.get("durationValue", 7))

        today = datetime.now()
        if duration_type == "days":
            start_date = today - timedelta(days=duration_value)
        elif duration_type == "weeks":
            start_date = today - timedelta(weeks=duration_value)
        elif duration_type == "months":
            start_date = today - timedelta(days=30 * duration_value)
        elif duration_type == "years":
            start_date = today - timedelta(days=365 * duration_value)
        elif duration_type == "qtr":
            start_date = today - timedelta(days=90 * duration_value)
        else:
            start_date = today - timedelta(days=7)  # default fallback

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = today.strftime("%Y-%m-%d")

        # Initialize clustering engine
        clustering = RestaurantReviewClustering(restaurant_name=restaurant)

        df = clustering.run_multi_aspect_clustering(
            aspects=["food", "service", "price", "ambiance", "others"],
            sentiment=sentiment,
            start_date=start_date_str,
            end_date=end_date_str,
            platform=platform,
            save_output=False,
            hash_inputs=False,
            parallel=True
        )

        if df.empty:
            return jsonify([])

        aspect_colors = {
            "food": "#D82E5E",
            "service": "#8725E0",
            "price": "#E8B501",
            "ambiance": "#D3642C",
            "others": "#8BC63E"
        }

        sankey_data = []
        root_node = sentiment.capitalize()

        for aspect, color in aspect_colors.items():
            col = f"sent_{aspect}_sentiment"
            if col in df.columns:
                count = df[df[col] == sentiment].shape[0]
                if count > 0:
                    sankey_data.append({
                        "from": root_node,
                        "to": aspect.capitalize(),
                        "value": count,
                        "color": color
                    })

        return jsonify(sankey_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Clustering API is running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
