import os
import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# -----------------------------
# Create visualisations folder
# -----------------------------
VIS_DIR = "./instagram_visualisations"
os.makedirs(VIS_DIR, exist_ok=True)

# -----------------------------
# 1. Load data
# -----------------------------
edges = pd.read_csv('./instagram_dataset/edges_clean.csv')
nodes = pd.read_csv('./instagram_dataset/nodes.csv')

### >>> Quick sanity check on data
print("Edges shape:", edges.shape)
print(edges.head(), "\n")

print("Nodes shape:", nodes.shape)
print(nodes.head(), "\n")

# -----------------------------
# 2. Build directed graph
# -----------------------------
G = nx.DiGraph()

for _, row in nodes.iterrows():
    G.add_node(
        row['id'],
        pos=row['pos'],
        flr=row['flr'],
        flg=row['flg'],
        eg=row['eg'],
        er=row['er'],
        fg=row['fg'],
        op=row['op']
    )

for _, row in edges.iterrows():
    G.add_edge(int(row['src']), int(row['dst']), weight=float(row['weight']))

### >>> Print basic graph info
print("Graph G info:")
print(f"Type: {type(G)}")
print(f"Directed: {G.is_directed()}")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

### >>> Peek at a few nodes and edges
print("\nSample nodes with attributes:")
for n, attrs in list(G.nodes(data=True))[:5]:
    print(f"Node {n}: {attrs}")

print("\nSample edges with weights:")
for u, v, w in list(G.edges(data=True))[:10]:
    print(f"{u} -> {v}, weight={w['weight']}")
print("\n")

# -----------------------------
# 3. Centrality measures
# -----------------------------
print("Computing in/out degree centrality...")
in_deg_c = nx.in_degree_centrality(G)
out_deg_c = nx.out_degree_centrality(G)
print("  -> Done degree centrality.")

print("Computing PageRank...")
pr = nx.pagerank(G, alpha=0.85, weight='weight')
print("  -> Done PageRank.")

print("Computing betweenness centrality (approx)...")
betw_c = nx.betweenness_centrality(
    G,
    k=200,                 # approx, adjust if needed
    normalized=True,
    weight='weight',
    seed=42
)
print("  -> Done betweenness centrality.")

# -----------------------------
# 4. Build centrality DataFrame + merge
# -----------------------------
cent_df = pd.DataFrame({
    'id': list(G.nodes())
})

cent_df['in_deg_c']  = cent_df['id'].map(in_deg_c)
cent_df['out_deg_c'] = cent_df['id'].map(out_deg_c)
cent_df['betw_c']    = cent_df['id'].map(betw_c)
cent_df['pr']        = cent_df['id'].map(pr)

### >>> Inspect centrality DataFrame
print("Centrality dataframe head:")
print(cent_df.head(), "\n")

# Merge with nodes (attributes)
data = cent_df.merge(nodes, left_on='id', right_on='id', how='inner')

print("Merged data shape:", data.shape)
print("Merged data head:")
print(data.head(), "\n")

# -----------------------------
# 5. Normalising centralities
# -----------------------------
features = data[['in_deg_c', 'betw_c', 'pr']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

data['in_deg_norm'] = features_scaled[:, 0]
data['betw_norm']   = features_scaled[:, 1]
data['pr_norm']     = features_scaled[:, 2]

### >>> Check normalized centralities
print("Normalized centrality columns head:")
print(data[['id', 'in_deg_norm', 'betw_norm', 'pr_norm']].head(), "\n")

# -----------------------------
# 6. Learning LAII (Layer-Aware Influencer Index) via regression
# -----------------------------
X = data[['in_deg_norm', 'betw_norm', 'pr_norm']]
y = data['er']

reg = LinearRegression()
reg.fit(X, y)

print("=== Linear Regression for LAII ===")
print("Coefficients (alpha, beta, gamma):", reg.coef_)
print("Intercept:", reg.intercept_)
print("R^2 on training:", reg.score(X, y))
print("=================================\n")

data['LAII'] = reg.predict(X)

### >>> Quick check of LAII distribution
print("LAII summary statistics:")
print(data['LAII'].describe(), "\n")

# -----------------------------
# 7. Correlations with er, fg, op
# -----------------------------
cols_to_compare = ['in_deg_c', 'betw_c', 'pr', 'LAII']
targets = ['er', 'fg', 'op']

for t in targets:
    print(f"\n=== Correlations with {t} ===")
    corr_series = data[cols_to_compare + [t]].corr()[t].sort_values(ascending=False)
    print(corr_series)
print("\n")

# -----------------------------
# 8. Top-by-followers vs top-by-LAII
# -----------------------------
top_by_flr = data.sort_values('flr', ascending=False).head(20)
print("Top 20 accounts by followers (flr):")
print(top_by_flr[['id', 'flr', 'er', 'fg', 'op', 'LAII']].reset_index(drop=True), "\n")

top_by_laii = data.sort_values('LAII', ascending=False).head(20)
print("Top 20 accounts by LAII:")
print(top_by_laii[['id', 'flr', 'er', 'fg', 'op', 'LAII']].reset_index(drop=True), "\n")

set_flr  = set(top_by_flr['id'])
set_laii = set(top_by_laii['id'])

print("Common in both top-20:", len(set_flr & set_laii))
print("Only in LAII top-20:", len(set_laii - set_flr))
print("Only in followers top-20:", len(set_flr - set_laii), "\n")

# -----------------------------
# 9. Visualisations (saved to ./instagram_visualisations)
# -----------------------------

# Scatter: in-degree vs LAII
plt.figure()
plt.scatter(data['in_deg_c'], data['LAII'], alpha=0.3)
plt.xlabel('In-degree centrality')
plt.ylabel('LAII (predicted engagement)')
plt.title('In-degree vs LAII')
plt.xscale('log')   # if needed
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "scatter_in_degree_vs_LAII.png"), dpi=300)
plt.close()

# Scatter: followers vs LAII
plt.figure()
plt.scatter(data['flr'], data['LAII'], alpha=0.3)
plt.xlabel('Followers (flr)')
plt.ylabel('LAII')
plt.title('Followers vs LAII')
plt.xscale('log')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "scatter_followers_vs_LAII.png"), dpi=300)
plt.close()

# -----------------------------
# 10. Community-level micro-influencers
# -----------------------------
print("Running community detection (Louvain)...")
H = G.to_undirected()
partition = community_louvain.best_partition(H)  # dict: node -> community id
print("Number of communities found:", len(set(partition.values())), "\n")

data['community'] = data['id'].map(partition)

# Top LAII in each community
top_comm = (
    data.sort_values('LAII', ascending=False)
         .groupby('community')
         .head(3)
)

print("Top LAII accounts in each community (up to 3 per community):")
print(
    top_comm[['community', 'id', 'flr', 'er', 'LAII']]
    .sort_values(['community', 'LAII'], ascending=[True, False])
    .head(50)      # just showing first 50 rows in case there are many communities
)
