import boto3
import pandas as pd
import json
import time
import random
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

# Initialize SQS client
sqs = boto3.client("sqs", region_name="us-west-1")
load_dotenv()
QUEUE_URL = os.getenv("SQS_QUEUE_URL")
if QUEUE_URL is None:
    raise ValueError("SQS_QUEUE_URL is not set in the environment variables.")

os.makedirs("data", exist_ok=True)
simulated_data_path = "data/real_time_simulated.csv"
simulated_data = []

# Load and preprocess data
df = pd.read_csv("data/raw/creditcard.csv")

# Drop 'Time' and 'Class' columns
df = df.drop(columns=["Time", "Class"])

# Scale 'Amount' column
if "Amount" in df.columns:
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Simulate real-time data messages
for i in range(50):
    row = df.sample(1).values.flatten().tolist()
    message = {"features": row}
    
    sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(message))
    simulated_data.append(row)
    print(f"[SENT] Message {i+1}")
    
    # Simulate varied message arrival times
    time.sleep(random.uniform(0.1, 0.5))

# Save collected features
pd.DataFrame(simulated_data, columns=df.columns).to_csv(simulated_data_path, index=False)
print(f"[INFO] Saved real-time simulated features to '{simulated_data_path}'")

baseline = pd.read_csv("data/processed/X_train.csv")
current = pd.read_csv("data/real_time_simulated.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=baseline, current_data=current)
report.save_html("results/drift_report.html")