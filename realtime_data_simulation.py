import boto3
import pandas as pd
import json
import time
import random
from sklearn.preprocessing import StandardScaler

# Initialize SQS client
sqs = boto3.client("sqs", region_name="us-west-1")
QUEUE_URL = "https://sqs.us-west-1.amazonaws.com/354918375950/fraud-detection"

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
    print(f"[SENT] Message {i+1}")
    
    # Simulate varied message arrival times
    time.sleep(random.uniform(0.1, 0.5))
