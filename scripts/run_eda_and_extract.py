#!/usr/bin/env python3
"""Run a quick EDA programmatically and dump key metrics to results.json.

This script mirrors the notebook's default behavior but runs headless and
saves a small JSON summary plus PNG figures under outputs/eda/.
"""
from pathlib import Path
import json
import time
import math
import sys

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="notebook")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "twitter"
OUT_DIR = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FILES_TO_LOAD = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TOP_K = 25

def read_edges_file(path: Path):
    edges = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u == v:
                continue
            edges.append((u, v))
    return edges

def build_graph_from_files(files):
    edges = []
    for p in files:
        edges.extend(read_edges_file(p))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

def main():
    files = sorted(DATA_DIR.glob("*.edges"))
    if not files:
        print("No .edges files found under", DATA_DIR)
        return 1
    files_to_load = files[:N_FILES_TO_LOAD]
    start = time.time()
    G = build_graph_from_files(files_to_load)
    elapsed = time.time() - start

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_in = sum(d for _, d in G.in_degree()) / max(1, n_nodes)
    avg_out = sum(d for _, d in G.out_degree()) / max(1, n_nodes)
    density = nx.density(G)

    # degree dataframe
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_df = pd.DataFrame({
        'node': list(G.nodes()),
        'in_degree': [in_deg.get(n, 0) for n in G.nodes()],
        'out_degree': [out_deg.get(n, 0) for n in G.nodes()],
    })

    # degree distributions plot
    plt.figure(figsize=(6,4))
    sns.histplot(deg_df['in_degree'], bins=50)
    plt.title('In-degree distribution')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'in_degree_hist.png', dpi=150)
    plt.close()

    # PageRank (use networkx implementation)
    try:
        pr = nx.pagerank(G, alpha=0.85)
    except Exception:
        # fallback simple power iteration
        nodes = list(G.nodes())
        N = len(nodes)
        pr = {n: 1.0/N for n in nodes}
        out_deg_map = dict(G.out_degree())
        dangling = [n for n,d in out_deg_map.items() if d==0]
        alpha = 0.85
        for _ in range(100):
            new_rank = {n: (1-alpha)/N for n in nodes}
            dangling_sum = alpha * sum(pr[n] for n in dangling) / N
            for n in nodes:
                new_rank[n] += dangling_sum
            for n in nodes:
                d = out_deg_map.get(n,0)
                if d==0: continue
                contrib = alpha * pr[n] / d
                for nbr in G.successors(n):
                    new_rank[nbr] += contrib
            diff = sum(abs(new_rank[n]-pr[n]) for n in nodes)
            pr = new_rank
            if diff < 1e-6:
                break
        s = sum(pr.values())
        if s>0:
            pr = {k:v/s for k,v in pr.items()}

    pr_df = pd.DataFrame.from_dict(pr, orient='index', columns=['pagerank']).reset_index().rename(columns={'index':'node'})
    merged = deg_df.merge(pr_df, on='node', how='left').fillna(0)

    # compute top-k sets and metrics
    top_in = merged.sort_values('in_degree', ascending=False).head(TOP_K)['node'].tolist()
    top_pr = merged.sort_values('pagerank', ascending=False).head(TOP_K)['node'].tolist()
    set_in = set(top_in)
    set_pr = set(top_pr)
    jaccard = len(set_in & set_pr) / len(set_in | set_pr) if (set_in | set_pr) else 0.0

    # spearman
    from scipy.stats import spearmanr
    merged['in_rank'] = merged['in_degree'].rank(ascending=False, method='dense')
    merged['pr_rank'] = merged['pagerank'].rank(ascending=False, method='dense')
    rho, pval = spearmanr(merged['in_rank'], merged['pr_rank'])

    results = {
        'files_loaded': len(files_to_load),
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'avg_in': float(avg_in),
        'avg_out': float(avg_out),
        'density': float(density),
        'read_time_s': elapsed,
        'topk_jaccard_in_pr': float(jaccard),
        'spearman_in_pr_rho': float(rho) if not np.isnan(rho) else None,
        'spearman_in_pr_pval': float(pval) if not np.isnan(pval) else None,
        'top_in_degree': top_in,
        'top_pagerank': top_pr,
    }

    # save merged full ranking to CSV
    merged[['node','in_degree','pagerank']].sort_values('pagerank', ascending=False).to_csv(OUT_DIR / 'full_ranking.csv', index=False)

    # save small rank scatter
    samp = merged.sample(frac=0.5, random_state=1) if len(merged) > 5000 else merged
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='in_rank', y='pr_rank', data=samp, alpha=0.6, s=20)
    plt.gca().invert_xaxis(); plt.gca().invert_yaxis()
    plt.xlabel('In-degree rank'); plt.ylabel('PageRank rank')
    plt.title('Rank scatter: in-degree vs PageRank')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'rank_scatter.png', dpi=150)
    plt.close()

    # try approximate betweenness if small
    if n_nodes <= 5000:
        k = min(200, n_nodes)
        bt = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
        bt_df = pd.DataFrame.from_dict(bt, orient='index', columns=['betweenness']).reset_index().rename(columns={'index':'node'})
        merged = merged.merge(bt_df, on='node', how='left').fillna(0)
        results['top_betweenness'] = merged.sort_values('betweenness', ascending=False).head(TOP_K)['node'].tolist()
    else:
        results['top_betweenness'] = []

    # write results
    with (OUT_DIR / 'results.json').open('w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)

    print('Wrote results to', OUT_DIR / 'results.json')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
