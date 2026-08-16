import pandas as pd
import sqlite3

df = pd.read_csv("data/processed/final_dataset.csv")
conn = sqlite3.connect("data/friction.db")
df.to_sql("complaints", conn, if_exists="replace", index=False)
conn.close()
print("Loaded into SQLite: data/friction.db")