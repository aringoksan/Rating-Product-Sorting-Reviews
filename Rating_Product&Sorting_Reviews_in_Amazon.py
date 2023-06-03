import pandas as pd
import math
import scipy.stats as st

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 500)

def time_based_weighted_average(df, w1=28, w2=26, w3=24):
    print(df.loc[df["day_diff"] <= 120, "overall"].mean(),
          df.loc[(df["day_diff"] > 120) & (df["day_diff"] <= 360), "overall"].mean(),
          df.loc[df["day_diff"] > 360, "overall"].mean())
    return df.loc[df["day_diff"] <= 120, "overall"].mean() * w1 / 100 + \
           df.loc[(df["day_diff"] > 120) & (df["day_diff"] <= 360), "overall"].mean() * w2 / 100 + \
           df.loc[df["day_diff"] > 360, "overall"].mean() * w3 / 100
def score_pos_neg_diff(up, down):
    return up - down
def score_avg_rating(up, down):
    if (up + down) == 0:
        return 0
    return up / (up + down)
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

reviews = pd.read_csv("amazon_review.csv")
print(reviews.head())
print(reviews.describe().T)
average_rating = reviews["overall"].mean()
time_weighted_rating = time_based_weighted_average(reviews, 50, 30, 20)
reviews["helpful_no"] = reviews["total_vote"] - reviews["helpful_yes"]

reviews["score_pos_neg_diff"] = reviews.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
reviews["score_avg_rating"] = reviews.apply(lambda x: score_avg_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
reviews["wilson_lower_bound"] = reviews.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


print(reviews.sort_values("wilson_lower_bound", ascending=False).head(5))