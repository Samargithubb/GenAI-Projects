import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import schedule
import time

cred = credentials.Certificate("credential_path")
firebase_admin.initialize_app(cred)

db = firestore.client()

def load_data():
    docs = db.collection('products').stream()
    data = [doc.to_dict() for doc in docs]

    df = pd.DataFrame(data)
    print(df.columns)

    df.to_csv('data/grocery_data.csv', index=False)
    print("Data loaded and saved to grocery_data.csv")

schedule.every().hour.at(":00").do(load_data)

while True:
    schedule.run_pending()
    time.sleep(60)
