import pandas as pd

# 1) Read the original TXT as whitespace-separated
df = pd.read_csv(
    './instagram_dataset/edges.txt',  # path to your txt file
    sep=r'\s+',                      # split on whitespace
    header=None,
    names=['src', 'dst', 'weight']
)

print(df.head())
print(df.dtypes)

# 2) Save a clean CSV version (optional but nice)
df.to_csv('./instagram_dataset/edges_clean.csv', index=False)
