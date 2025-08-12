import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import struct
from datetime import datetime, timedelta

df = pd.read_csv("docs/SuperStoreOrders.csv", usecols = ["order_date", "sales"]);

df["order_date"] = df["order_date"].str.replace(r"[/-]", "-", regex=True);
df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce");

df["sales"] = df["sales"].astype(str).str.replace(",", "");
df["sales"] = pd.to_numeric(df["sales"], errors="coerce");

daily_sales = df.groupby("order_date")["sales"].sum();
second_half = daily_sales.iloc[(len(daily_sales) // 2):];
smooth = second_half.rolling(7).mean();

times = []
values = []

with open("docs/predicted.bin", "rb") as f:
    for i in range(60, 723):
        times.append(i);
        f.seek((i - 60) * 4);
        data = f.read(4);

        value = struct.unpack("f", data)[0];
        values.append(value);

start_date = datetime(2011, 1, 1);
dates = [start_date + timedelta(days=1461 / 2 + t) for t in times];

series = pd.Series(values, index=dates);
smooth2 = series.rolling(7).mean();

plt.figure(figsize=(14, 6));
plt.plot(smooth.index, smooth.values, label="Daily Sailes", color="blue");
plt.plot(dates, smooth2, color="green");
plt.title("Daily Sales Over Time");
plt.xlabel("Date");
plt.ylabel("Sales");
plt.grid(True);
plt.legend();
plt.tight_layout();
plt.show();
