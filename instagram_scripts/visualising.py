import networkx as nx
import pandas as pd

df = pd.read_csv('./instagram_dataset/edges_clean.csv')

G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['src'], row['dst'], weight=row['weight'])

print(G)