import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("docs/SuperStoreOrders.csv", usecols = ["order_date", "sales"]);

df["order_date"] = df["order_date"].str.replace(r"[/-]", "-", regex=True);
df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce");

df["sales"] = df["sales"].astype(str).str.replace(",", "");
df["sales"] = pd.to_numeric(df["sales"], errors="coerce");

daily_sales = df.groupby("order_date")["sales"].sum();
first_half = daily_sales.iloc[:(len(daily_sales) // 2)];
smooth = first_half.rolling(7).mean();

plt.figure(figsize=(14, 6));
plt.plot(smooth.index, smooth.values, label="Daily Sailes", color="blue");
plt.title("Daily Sales Over Time");
plt.xlabel("Date");
plt.ylabel("Sales");
plt.grid(True);
plt.legend();
plt.tight_layout();
plt.show();
