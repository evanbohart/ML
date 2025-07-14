import pandas as pd
import struct

df = pd.read_csv("docs/SuperStoreOrders.csv", usecols = ["order_date", "profit", "discount"]);

df["order_date"] = df["order_date"].str.replace(r"[/-]", "-", regex=True);
df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce");

daily_profits = df.groupby("order_date")["profit"].sum();
promotion_flags = df.groupby("order_date")["discount"].apply(lambda x: (x > 0).any()).astype(int);

full_range = pd.date_range(start=daily_profits.index.min(), end=daily_profits.index.max());

daily_df = pd.DataFrame(index=full_range);
daily_df["profit"] = daily_profits.reindex(full_range, fill_value=0.0);
daily_df["is_promotion"] = promotion_flags.reindex(full_range, fill_value=0);

daily_df["day_of_week"] = daily_df.index.dayofweek;
daily_df["month"] = daily_df.index.month - 1;
daily_df["is_weekend"] = daily_df["day_of_week"].isin([5, 6]).astype(int);

with open("docs/profits.bin", "wb") as f:
    for day, (date, row) in enumerate(daily_df.iterrows(), start=1):
        data = struct.pack(
            "ifiiii",
            int(day),
            float(row["profit"]),
            int(row["day_of_week"]),
            int(row["month"]),
            int(row["is_weekend"]),
            int(row["is_promotion"])
        );

        f.write(data);
