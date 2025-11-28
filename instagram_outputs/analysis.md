# Instagram Influence Analysis using LAII

## 1. Defining LAII (Layer-Aware Influence Index)

The **Layer-Aware Influence Index (LAII)** is a **learned influence score** that combines multiple network centrality measures into a single value designed to approximate a user’s **engagement efficiency**.

Formally, for each user \(v\) we compute:

- \( \text{in\_deg\_norm}(v) \): normalized in-degree centrality  
- \( \text{betw\_norm}(v) \): normalized betweenness centrality  
- \( \text{pr\_norm}(v) \): normalized PageRank

We then fit a **linear regression** model to predict the user’s **engagement rate** (`er`) from these features:

\[
\widehat{er}(v) = \alpha \cdot \text{in\_deg\_norm}(v)
\;+\;
\beta \cdot \text{betw\_norm}(v)
\;+\;
\gamma \cdot \text{pr\_norm}(v)
\;+\;
\delta
\]

The **LAII score** is defined as this predicted engagement rate:

\[
\text{LAII}(v) = \widehat{er}(v)
\]

Intuitively:

- It is **not** just follower count or one centrality.
- It is a **data-driven combination** of structural signals, tuned to best match actual engagement behaviour.
- High LAII ≈ “given where you sit in the network, you look like a **high-engagement influencer**”.

---

## 2. Dataset and Graph Construction

- Nodes = **70,409 Instagram accounts**
- Edges = **1,031,348 directed edges**  
  - Each edge `src → dst` represents a **follow** relation with an associated weight.
- Node attributes (from `nodes.csv`):
  - `flr`: followers  
  - `flg`: following  
  - `er`: engagement rate (likes + comments / followers)  
  - `eg`: engagement grade (1–12)  
  - `fg`: follower growth % over a month  
  - `op`: outsider percentage (non-followers who liked posts)  
  - `pos`: number of posts

We build a **directed follower graph** `G` in NetworkX:

- `G` is a `DiGraph`
- ~70k nodes, ~1M edges
- Each node carries its full attribute vector; each edge carries a weight.

---

## 3. Centrality-Based Features

On this directed graph, we compute:

1. **In-degree centrality** (`in_deg_c`)  
   - Popularity as “how many users follow you”.

2. **Out-degree centrality** (`out_deg_c`)  
   - How many accounts you follow (used mainly for context).

3. **Betweenness centrality** (`betw_c`) – approximate  
   - Using `nx.betweenness_centrality(G, k=200, weight='weight')`.  
   - Measures how often a node lies on shortest paths: **broker / bridge role**.

4. **PageRank** (`pr`) – weighted  
   - Using `nx.pagerank(G, alpha=0.85, weight='weight')`.  
   - Measures **global importance** based on incoming links from other important nodes.

We then create a combined DataFrame where each row (user) has:

- Centralities: `in_deg_c`, `out_deg_c`, `betw_c`, `pr`
- Attributes: `flr`, `flg`, `er`, `eg`, `fg`, `op`, `pos`

---

## 4. Normalisation and Learning LAII

To make centralities comparable, we standardise them:

- Use `StandardScaler` on `[in_deg_c, betw_c, pr]`
- Obtain:
  - `in_deg_norm`
  - `betw_norm`
  - `pr_norm`

We then fit a **linear regression model**:

- Features \(X = [\text{in\_deg\_norm}, \text{betw\_norm}, \text{pr\_norm}]\)
- Target \(y = er\) (engagement rate)

Resulting model (on this dataset):

- Coefficients:
  - \(\alpha\) (in-degree) ≈ **−2.61**
  - \(\beta\) (betweenness) ≈ **+1.29**
  - \(\gamma\) (PageRank) ≈ **−0.31**
- Intercept \(\delta\) ≈ **13.08**
- Training \(R^2\) ≈ **0.039** (about 4% of variance in `er` explained)

So:

\[
\text{LAII}(v) =
-2.61 \cdot \text{in\_deg\_norm}(v)
+1.29 \cdot \text{betw\_norm}(v)
-0.31 \cdot \text{pr\_norm}(v)
+13.08
\]

Interpretation:

- **Higher in-degree** → **lower** LAII (big hubs tend to have lower engagement per follower).  
- **Higher betweenness** → **higher** LAII (bridge roles support better engagement).  
- **Higher PageRank** → slight negative effect (very globally central nodes also show diluted engagement).

LAII is stored as `data['LAII']` and treated as a **structural proxy for engagement efficiency**.

---

## 5. Correlation Results

We compare how raw centralities and LAII relate to:

- `er` – engagement rate  
- `fg` – follower growth %  
- `op` – outsider percentage %

### 5.1 Correlation with Engagement Rate (`er`)

From the correlation matrix:

- `LAII` vs `er`: **+0.198**
- `in_deg_c` vs `er`: **−0.175**
- `pr` vs `er`: **−0.118**
- `betw_c` vs `er`: **−0.032**

Key observations:

- **All raw centralities have slightly negative correlation with engagement rate.**
  - Being a big hub or highly ranked by PageRank does **not** imply better engagement per follower.
- **LAII** is the only one with a **positive and stronger** correlation.
  - This confirms that a **learned combination** of centralities aligns better with actual engagement than any single metric.

### 5.2 Correlation with Follower Growth (`fg`) and Outsiders Percentage (`op`)

- Correlations with `fg` and `op` are small for all measures (including LAII).
- This is expected: growth and outsider reach depend strongly on **content quality, trends, and campaigns**, not just network position.

Interpretation:

> Network structure alone can’t fully predict growth or external reach, but LAII still performs at least as well as individual centralities and slightly better for growth.

---

## 6. Followers vs LAII

The **Followers vs LAII** scatter plot (log-scale on followers) shows:

- A dense horizontal band around **LAII ≈ 12–16** spanning almost the entire follower range.
- A **deep “valley” of very negative LAII values** (down to around −80) centred on mid-to-high follower counts (≈ \(10^2\)–\(10^4\) followers).
- The **highest LAII values (40–60)** correspond to accounts with **moderate follower counts** (a few thousand to tens of thousands), not the largest celebrities.

Conclusions:

1. **Follower count is a poor proxy for influence efficiency.**  
   LAII does not systematically increase with followers; high and low LAII values appear at many follower levels.

2. The **negative valley** around mid–high followers represents **inefficient large accounts**:
   - They look structurally important (many followers, high centrality),
   - But the learned model predicts **very low engagement per follower**.

3. The **top LAII outliers** are **micro-influencers**:
   - Mid-sized accounts with strong engagement and favourable structural positions.
   - These are exactly the users that brands often want for targeted campaigns.

---

## 7. In-Degree vs LAII

The **In-degree vs LAII** scatter plot (log-scale on in-degree centrality) shows:

- For small in-degree, LAII values cluster around ~12–16.
- As in-degree grows large, **LAII tends to decrease**, with many high in-degree nodes having strongly negative LAII.

This visually reinforces:

- The **negative regression coefficient** on in-degree, and  
- The **negative correlation** between in-degree and engagement rate.

Interpretation:

> “Being followed by many accounts does not guarantee good engagement. In fact, very popular nodes often have lower engagement efficiency, so LAII penalizes them. The most efficient influencers are not the biggest hubs, but those with strategic positions and good brokerage.”

This aligns with known phenomena such as **engagement dilution** for very large accounts.

---

## 8. Ranking: Top by Followers vs Top by LAII

We compared top-20 users under two ranking schemes:

- **Top-20 by followers (`flr`)**
  - All are **very large accounts** (≈ 175k–1.3M followers).
  - Engagement rates (`er`) are typically moderate (low single digits).

- **Top-20 by LAII**
  - Followers mostly in the **hundreds to low tens of thousands**.
  - Many accounts show **very high engagement rates**, e.g.:
    - id 48692: 1,700 followers, `er ≈ 27.35`, high LAII  
    - id 42974: 646 followers, `er ≈ 17.00`, high LAII  
    - Several others with `er` in the teens or above.

- **Overlap:**
  - **0 accounts in common** between the top-20 by followers and top-20 by LAII.

This is a crucial result:

> “Follower-based ranking primarily surfaces mega-accounts with average engagement. LAII ranking surfaces a completely different set of **micro-influencers**: users with modest follower counts but exceptional engagement efficiency. There is zero overlap between the top-20 by followers and top-20 by LAII, highlighting how different these two notions of ‘influence’ really are.”

---

## 9. Community-Level Micro-Influencers (Louvain)

We run **Louvain community detection** on the undirected version of the follower graph:

- Number of communities found: **165**

For each community, we list the **top-3 users by LAII**.

Patterns observed:

- Many top community members have:
  - Followers in the **hundreds or low thousands**,
  - **Very high engagement rates** (often >5, sometimes >15 or even ~50),
  - High LAII scores.

- In some communities, a large global account appears among the top-3, but is often **second or third** behind a smaller local account with higher LAII.

Example patterns:

- Community 3:
  - id 778: 4,100 followers, LAII ≈ 66  
  - id 5095: 17,400 followers, LAII ≈ 63  
  - id 529: 1,100 followers, LAII ≈ 44  
  All three are strong micro-influencers within this community.

- Other communities show similar behaviour: local champions with modest followers but very strong predicted engagement.

Interpretation:

> LAII is able to identify **community-level micro-influencers** – accounts that may not dominate the global network, but are key influencers inside their own niche clusters. These are especially valuable for **targeted marketing and community-focused campaigns**.

---

## 10. Overall Conclusions

1. **Network structure alone does not guarantee high engagement,** and raw centralities can be misleading:
   - High in-degree and high PageRank often correlate **negatively** with engagement rate.
   - Simply picking the biggest or most central accounts is not an effective way to choose influencers.

2. The **Layer-Aware Influence Index (LAII)** provides a more meaningful influence measure:
   - It is a learned combination of in-degree, betweenness, and PageRank tuned to predict engagement rate.
   - LAII shows a **positive and stronger correlation** with engagement than any single centrality.

3. **Follower count vs LAII tells two very different stories:**
   - Follower-based ranking surfaces global celebrities with average engagement.
   - LAII ranking surfaces mid-sized accounts with outstanding engagement efficiency – the **micro-influencers**.

4. At the **community level**, LAII highlights **local champions**:
   - Within each of the 165 communities, LAII identifies a small set of high-impact nodes that are structurally and behaviourally influential inside their niche.

5. For **practical influencer selection**:
   - LAII is better suited than follower count or any single centrality when the goal is to find **efficient, community-embedded influencers** rather than just the largest accounts.

In summary, this analysis demonstrates that combining multiple centrality measures into a learned influence index (LAII), and validating it against real engagement metrics, yields a richer and more realistic view of influence in an Instagram network than traditional, single-measure approaches.
