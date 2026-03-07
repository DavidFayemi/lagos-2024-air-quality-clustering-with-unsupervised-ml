import os
import sys
import json
import tempfile
import papermill as pm
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import date, timedelta, datetime
from pymongo import MongoClient, ReplaceOne

# Load secrets from .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME", "lagos_air_quality")

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in .env")

# Resolve month label
if len(sys.argv) > 1:
    MONTH_LABEL = sys.argv[1]
else:
    first_of_this_month = date.today().replace(day=1)
    last_month = first_of_this_month - timedelta(days=1)
    MONTH_LABEL = last_month.strftime("%B %Y")

print(f"► Target month: {MONTH_LABEL}")

# Fetch CSV from open.africa
DATASET_URL = "https://open.africa/eu/dataset/sensorsafrica-airquality-archive-lagos"

print(f"► Fetching index: {DATASET_URL}")
resp = requests.get(DATASET_URL, timeout=30)
resp.raise_for_status()

soup  = BeautifulSoup(resp.text, "html.parser")
items = soup.select("li.resource-item")

matches = []
for item in items:
    heading = item.select_one("a.heading")
    if heading and MONTH_LABEL in heading.get_text():
        dl_link = item.select_one("a.resource-url-analytics")
        if dl_link:
            matches.append({
                "title": heading.get_text(strip=True),
                "url":   dl_link["href"],
            })

if not matches:
    raise FileNotFoundError(
        f"No resource found on the portal for '{MONTH_LABEL}'. "
        "Data may not be published yet — retry later."
    )

target = matches[0]
print(f"► Found  : {target['title']}")
print(f"► Downloading: {target['url']}")

dl = requests.get(target["url"], timeout=60)
dl.raise_for_status()
print(f"✓ Download complete")

# Run notebook, read results, save to MongoDB 
tmp_csv     = tempfile.NamedTemporaryFile(suffix=".csv",  delete=False)
tmp_results = tempfile.NamedTemporaryFile(suffix=".json", delete=False)

try:
    # Write downloaded CSV to temp file
    tmp_csv.write(dl.content)
    tmp_csv.close()
    tmp_results.close()   # just need the path, notebook writes to it

    print(f"► Temp CSV     : {tmp_csv.name}")
    print(f"► Temp results : {tmp_results.name}")

    # Run notebook
    os.makedirs("outputs", exist_ok=True)
    pm.execute_notebook(
        input_path="Lagos(2024) Air quality clustering using unsupervised machine learning.ipynb",
        output_path=f"outputs/Lagos(2024) Air quality clustering using unsupervised machine learning_{MONTH_LABEL.replace(' ', '_')}.ipynb",
        parameters={
            "MONTH_LABEL":  MONTH_LABEL,
            "DATA_PATH":    tmp_csv.name,
            "RESULTS_PATH": tmp_results.name,
        }
    )
    print(f"✓ Notebook execution complete")

    # Read results JSON written by notebook
    with open(tmp_results.name, "r") as f:
        results = json.load(f)

    # Save to MongoDB 
    parsed    = datetime.strptime(MONTH_LABEL, "%B %Y")
    month_id  = f"{parsed.year}-{parsed.month:02d}"

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    # upsert processed_months document
    processed_month_doc = {
        "_id":          month_id,
        "year":         parsed.year,
        "month":        parsed.month,
        "label":        MONTH_LABEL,
        "status":       "completed",
        "processed_at": datetime.utcnow(),
        "kmeans":       results["kmeans"],
        "dbscan":       results["dbscan"],
        "hierarchical": results["hierarchical"],
    }

    db.processed_months.replace_one(
        {"_id": month_id},
        processed_month_doc,
        upsert=True
    )
    print(f"✓ Upserted processed_months document for {MONTH_LABEL}")

    # Bulk upsert daily_observations
    ops = [
        ReplaceOne(
            {"_id": f"{month_id}_{obs['sensor_id']}_{obs['date']}"},
            {"_id": f"{month_id}_{obs['sensor_id']}_{obs['date']}", "month_id": month_id, **obs},
            upsert=True
        )
        for obs in results["observations"]
    ]
    result = db.daily_observations.bulk_write(ops)
    print(f"✓ Upserted {result.upserted_count} new + {result.modified_count} updated daily_observations")

    client.close()
    print(f"\n✓ All done — {MONTH_LABEL} saved to MongoDB")

finally:
    os.unlink(tmp_csv.name)
    os.unlink(tmp_results.name)
    print(f"✓ Temp files cleaned up")