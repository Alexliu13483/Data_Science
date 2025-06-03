import pandas as pd
reviews = pd.read_csv("../datasets/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

reviews.groupby(['country']).price.agg([len, min, max])
