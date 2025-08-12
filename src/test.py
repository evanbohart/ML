import pandas as pd
import struct
from datetime import timedelta

df = pd.read_csv("docs/SuperStoreOrders.csv", usecols = ["order_date", "sales", "discount"]);

df["order_date"] = df["order_date"].str.replace(r"[/-]", "-", regex=True);
df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce");

df["sales"] = df["sales"].astype(str).str.replace(",", "");
df["sales"] = pd.to_numeric(df["sales"], errors="coerce");

daily_sales = df.groupby("order_date")["sales"].sum();
second_half = daily_sales.iloc[(len(daily_sales) // 2):];
smooth = second_half.rolling(7, min_periods=1).mean();

full_range = pd.date_range(start=smooth.index.min(), end=smooth.index.max());
daily_df = pd.DataFrame(index=full_range);
daily_df["sales"] = second_half.reindex(full_range, fill_value=0.0);
daily_df["avg_sales"] = smooth.reindex(full_range, fill_value=0.0);

promotion_flags = df.groupby("order_date")["discount"].apply(lambda x: (x > 0).any()).astype(int);

daily_df["is_promotion"] = promotion_flags.reindex(full_range, fill_value=0);
daily_df["day_of_week"] = daily_df.index.dayofweek;
daily_df["month"] = daily_df.index.month - 1;
daily_df["is_weekend"] = daily_df["day_of_week"].isin([5, 6]).astype(int);

holidays = [
    "1-1-2011",
    "4-7-2011",
    "24-11-2011",
    "25-12-2011",
    "1-1-2012",
    "4-7-2012",
    "22-11-2012"
    "25-12-2012",
    "1-1-2013",
    "4-7-2013",
    "28-11-2013"
    "25-12-2013",
    "1-1-2014",
    "4-7-2014",
    "27-11-2014",
    "25-12-2014"
]

holiday_flags = set();

for h in holidays:
    date = pd.to_datetime(h, format="%d-%m-%Y", errors="coerce");
    for offset in range(-14, 0):
        holiday_flags.add(date + timedelta(days=offset));

daily_df["is_holiday"] = daily_df.index.isin(holiday_flags).astype(int)

count = 0;
with open("docs/second_half_sales.bin", "wb") as f:
    for day, (date, row) in enumerate(daily_df.iterrows(), start=1):
        data = struct.pack(
            "iffiiiii",
            int(day),
            float(row["sales"]),
            float(row["avg_sales"]),
            int(row["day_of_week"]),
            int(row["month"]),
            int(row["is_weekend"]),
            int(row["is_promotion"]),
            int(row["is_holiday"])
        );

        f.write(data);

        print(count);
        count += 1
