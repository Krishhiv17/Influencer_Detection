# Influencer Detection — SNAP Twitter centrality comparison

This repository computes and compares centrality measures (in-degree, approximate betweenness, PageRank)
on the SNAP Twitter ego-networks dataset to identify potential influencers.

## What this repo contains
- `analyze_twitter_centrality.py` — main script that merges `.edges` files into a directed follower graph and
  computes in-degree, approximate betweenness, and PageRank rankings.
- `twitter/` — directory with SNAP `.edges` and related files (already present in the repo).
- `outputs/` — past run outputs (top-k lists, experiment artifacts).

## Prerequisites
- Python 3.9+ recommended
- Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic usage
Place the SNAP `.edges` files in the `twitter/` directory (this repo already contains them).

Run the script with defaults:

```bash
python analyze_twitter_centrality.py
```

Example with options:

```bash
python analyze_twitter_centrality.py --data-dir twitter --top-k 25 --pagerank-alpha 0.85
```

Key CLI flags:
- `--data-dir` — path to the folder with `.edges` files (default: `twitter/`).
- `--top-k` — number of top users to display for each metric.
- `--betweenness-samples` and `--betweenness-seed` — tune approximate betweenness.
- `--pagerank-alpha` / `--pagerank-tol` / `--pagerank-max-iter` — PageRank params.



