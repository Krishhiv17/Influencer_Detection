"""Compute centrality rankings for the SNAP Twitter ego network dataset."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import networkx as nx


def load_follower_graph(data_dir: Path) -> nx.DiGraph:
    """Build a directed follower graph by merging all .edges files."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    graph = nx.DiGraph()
    edge_files = sorted(data_dir.glob("*.edges"))

    if not edge_files:
        raise FileNotFoundError(f"No .edges files found under {data_dir}")

    for edge_file in edge_files:
        with edge_file.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                follower, followee = parts[0], parts[1]
                if follower == followee:
                    continue

                graph.add_edge(follower, followee)

    return graph


def top_n(
    values: Dict[str, float],
    n: int,
) -> Iterable[Tuple[str, float]]:
    """Yield the top-n items from a score mapping."""
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:n]


def summarize_overlap(rankings: Dict[str, Iterable[str]]) -> Dict[str, Counter]:
    """Compute overlaps between ranked node lists."""
    metrics = list(rankings)
    summary: Dict[str, Counter] = {}

    for metric in metrics:
        shared = Counter()
        metric_nodes = set(rankings[metric])
        for other_metric in metrics:
            if other_metric == metric:
                continue
            shared[other_metric] = len(metric_nodes.intersection(rankings[other_metric]))
        summary[metric] = shared

    return summary


def pagerank_power_iteration(
    graph: nx.DiGraph,
    alpha: float,
    max_iter: int,
    tol: float,
) -> Dict[str, float]:
    """Pure Python PageRank power-iteration to avoid SciPy dependency issues."""
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return {}

    rank = {node: 1.0 / node_count for node in graph}
    out_degree = dict(graph.out_degree())
    dangling_nodes = [node for node, degree in out_degree.items() if degree == 0]
    min_value = (1.0 - alpha) / node_count

    for _ in range(max_iter):
        new_rank = {node: min_value for node in rank}
        dangling_contrib = alpha * sum(rank[node] for node in dangling_nodes) / node_count
        for node in new_rank:
            new_rank[node] += dangling_contrib

        for node, neighbors in graph.adjacency():
            degree = out_degree[node]
            if degree == 0:
                continue
            spread = alpha * rank[node] / degree
            for neighbor in neighbors:
                new_rank[neighbor] += spread

        diff = sum(abs(new_rank[node] - rank[node]) for node in rank)
        rank = new_rank
        if diff < tol:
            break

    return rank


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute degree, betweenness (approximate), and PageRank centrality "
            "rankings for the SNAP Twitter social circles dataset."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "twitter",
        help="Path to the directory containing *.edges files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top users to display per centrality metric.",
    )
    parser.add_argument(
        "--betweenness-samples",
        type=int,
        default=200,
        help="Sample size for approximate betweenness centrality.",
    )
    parser.add_argument(
        "--betweenness-seed",
        type=int,
        default=42,
        help="Random seed for betweenness sampling.",
    )
    parser.add_argument(
        "--pagerank-alpha",
        type=float,
        default=0.85,
        help="Damping factor (alpha) for PageRank.",
    )
    parser.add_argument(
        "--pagerank-tol",
        type=float,
        default=1.0e-06,
        help="Tolerance for PageRank power iteration convergence.",
    )
    parser.add_argument(
        "--pagerank-max-iter",
        type=int,
        default=100,
        help="Maximum iterations for PageRank power iteration.",
    )
    args = parser.parse_args()

    try:
        graph = load_follower_graph(args.data_dir)
    except FileNotFoundError as err:
        parser.error(str(err))

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    print(f"Loaded follower graph with {node_count:,} users and {edge_count:,} directed edges.\n")

    print("Computing centrality measures ...")
    in_degree_centrality = nx.in_degree_centrality(graph)
    betweenness_kwargs = {
        "k": min(args.betweenness_samples, node_count),
        "normalized": True,
        "seed": args.betweenness_seed,
    }
    try:
        betweenness_centrality = nx.betweenness_centrality(
            graph,
            backend="python",
            **betweenness_kwargs,
        )
    except (ImportError, TypeError):
        betweenness_centrality = nx.betweenness_centrality(
            graph,
            **betweenness_kwargs,
        )
    pagerank_scores = pagerank_power_iteration(
        graph,
        alpha=args.pagerank_alpha,
        max_iter=args.pagerank_max_iter,
        tol=args.pagerank_tol,
    )

    top_in_degree = list(top_n(in_degree_centrality, args.top_k))
    top_betweenness = list(top_n(betweenness_centrality, args.top_k))
    top_pagerank = list(top_n(pagerank_scores, args.top_k))

    rankings = {
        "in_degree": [node for node, _ in top_in_degree],
        "betweenness": [node for node, _ in top_betweenness],
        "pagerank": [node for node, _ in top_pagerank],
    }

    def display_block(title: str, entries: Iterable[Tuple[str, float]]) -> None:
        print(title)
        for rank, (node, score) in enumerate(entries, start=1):
            follower_count = graph.in_degree(node)
            print(f"  {rank:2d}. user {node} -> score={score:.6f} followers={follower_count}")
        print()

    display_block("Top users by in-degree centrality (followers):", top_in_degree)
    display_block("Top users by betweenness centrality (bridges):", top_betweenness)
    display_block("Top users by PageRank centrality (influence):", top_pagerank)

    overlaps = summarize_overlap(rankings)
    print("Overlap in top lists (counts of shared users):")
    for metric, shared_counts in overlaps.items():
        parts = ", ".join(f"{other}: {shared_counts[other]}" for other in rankings if other != metric)
        print(f"  {metric} shares {parts}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")
