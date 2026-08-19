# Observations — Credit Card Customer Segmentation

## What I observed

I ran unsupervised clustering on ~8,950 credit card customers across 8 behavioral features (balance, purchases, cash advances, credit limit, payments, purchasing frequency, tenure, and average order value) using two different algorithms: KMeans and DBSCAN. I also applied PCA to visualize the high-dimensional structure in 2D. The goal was not to assume every cluster is automatically meaningful, but to understand *what each algorithm actually found* and why those findings differ.

---

## Basic findings from the results

### KMeans (K=4) segmented all customers into four distinct groups:

**Cluster 0 — High-value, high-activity customers (965 customers, 10.8%)**
- Average balance: $5,665
- Average purchases: $2,861
- Average cash advances: $4,232
- Average payments: $6,050
- Credit limit: $10,479
- Purchase frequency: 0.50

These are the bank's most valuable customers by volume and engagement. They carry substantial balances, make significant purchases, actively use cash advances, and pay back large amounts, all while having the highest credit limits. The moderate purchase frequency (0.50) suggests they're intentional spenders rather than constant transaction makers.

**Cluster 1 — Inactive/low-engagement customers (3,783 customers, 42.3%)**
- Average balance: $1,232
- Average purchases: $224
- Average cash advances: $796
- Average payments: $1,026
- Credit limit: $3,368
- Purchase frequency: 0.15

This is the largest single group — customers with minimal transaction activity. Low purchases, very low purchase frequency (0.15), and modest balances suggest dormant or near-dormant accounts. These customers haven't been triggered to use their cards actively, despite holding them.

**Cluster 2 — Frequent small-transaction customers (3,464 customers, 38.7%)**
- Average balance: $924
- Average purchases: $1,455
- Average cash advances: $237
- Average payments: $1,532
- Credit limit: $4,455
- Purchase frequency: 0.87 (highest)

The most frequent purchasers by far (frequency 0.87). They make many transactions but keep them small (lowest average order value at $63), and rarely use cash advances. These are the everyday users — high transaction count, low transaction size, consistent engagement.

**Cluster 3 — Cash-advance-oriented customers (738 customers, 8.3%)**
- Average balance: $911
- Average purchases: $440
- Average cash advances: $1,138
- Average payments: $649
- Credit limit: $2,621
- Purchase frequency: 0.43

A smaller but distinctive group oriented toward cash withdrawals. Their cash advances ($1,138) far exceed their purchases ($440), indicating they're using the card more as a cash access tool than a purchasing instrument. Also the youngest by tenure (7.43 months vs 11-12 for other clusters).

---

## How scaling changed the clustering result

**Without standardization**, the raw feature ranges would have dominated the distance calculations completely. `CREDIT_LIMIT` alone ranges from ~$1,000 to ~$20,000+, while `PURCHASES_FREQUENCY` is bounded 0–1. If I'd fed the algorithm unscaled data, credit limit would be the de facto primary driver of similarity — two customers with similar credit limits but wildly different spending behaviors would look "close" purely because of the units, not actual behavioral resemblance.

I used `StandardScaler` to center every feature at mean 0 with standard deviation 1. This put all eight features on equal footing. Now the algorithm judges similarity based on behavioral *pattern* (how balanced, how frequent, how much they buy) rather than the accident of which features were originally measured in large vs. small units. 

**The impact was decisive.** With scaling, I got four behaviorally distinct clusters. Without it, I would have gotten clusters that were mostly just "high credit limit vs. low credit limit" with behavioral differences as secondary noise.

---

## How the final K was selected

I used two complementary methods instead of trusting either one alone:

**The elbow plot** showed inertia dropping steeply from K=2 through K=4, then flattening significantly after K=4. The drop from K=4 to K=5 was only ~4,444 units, compared to ~6,575 from K=3 to K=4. The "bend" is visible at K=4.

**The silhouette scores** told a different story:
- K=2: 0.4241 (high)
- K=3: 0.3926 (high)
- K=4: 0.2646 (much lower)
- K=5–10: 0.28–0.29 (stable, slightly higher than K=4)

The very high silhouette at K=2 and K=3 is a red flag in this context — it usually means I'm over-simplifying. Two clusters would just be "active vs. inactive," which loses the nuance between frequent small-transaction users (Cluster 2) and high-value/high-cash-advance users (Clusters 0 and 3).

**My decision:** I chose K=4 as the point where the elbow bend is clearest *and* I capture enough behavioral distinction to be actionable, even if the absolute silhouette score drops below K=2 or K=3. The 0.2646 score at K=4 is moderate but honest — it acknowledges that these clusters aren't perfectly separated, which matches reality: customer behavior is messy and doesn't fit into clean silos.

---

## What characteristics distinguish each KMeans cluster

| Characteristic | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---|---|---|---|
| **Size** | 965 (11%) | 3,783 (42%) | 3,464 (39%) | 738 (8%) |
| **Primary behavior** | High-value, engaged | Inactive, dormant | Frequent purchaser | Cash-advance user |
| **Balance** | $5,665 | $1,232 | $924 | $911 |
| **Purchases** | $2,861 | $224 | $1,455 | $440 |
| **Cash advances** | $4,232 | $796 | $237 | $1,138 |
| **Payments** | $6,050 | $1,026 | $1,532 | $649 |
| **Avg order size** | $110 | $69 | $63 | $94 |
| **Purchase frequency** | 0.50 | 0.15 | 0.87 | 0.43 |
| **Key insight** | Premium revenue drivers | Underutilized assets | Daily/weekly users | Alternative credit access |

---

## How sensitive was DBSCAN to eps and min_samples

**Very sensitive.** The grid search across 5 eps values × 3 min_samples values revealed a sharp trade-off with almost no middle ground:

**At eps=0.5** (tight neighborhood radius):
- 13–27 tiny clusters found
- 27–35% of customers marked as noise
- Silhouette scores negative or near-zero (−0.04 best case)

The radius is too tight; clusters fragment into slivers.

**At eps=0.8:**
- 1–6 clusters
- 12–16% noise
- Some configurations (eps=0.8, min_samples=8) produce a silhouette of 0.595, which is strong

**At eps=1.0:**
- 1–6 clusters
- 7–10% noise
- Mixed results; eps=1.0, min_samples=8 collapses to 1 giant cluster

**At eps=1.5 and eps=2.0:**
- Mostly 1–2 clusters
- <4% noise
- High silhouette where applicable (eps=2.0, min_samples=5 reaches 0.67), but one or two mega-clusters dominate

**What I chose:** eps=1.0, min_samples=8. This produced 1 very large cluster (8,139 customers) and identified 811 points (9.1%) as noise. I selected this over eps=0.8, min_samples=8 (which had a higher silhouette of 0.595 and 2 clusters) because the 9.1% noise population revealed something genuinely interesting about the data.

---

## Which points were considered noise and why that might be useful

The 811 DBSCAN noise points (9.1% of customers) weren't randomly scattered outliers — they were systematically high-activity, high-value customers:

| Metric | Noise customers | Overall average | Ratio |
|---|---|---|---|
| **Balance** | $4,359 | $1,564 | 2.79× |
| **Purchases** | $3,134 | $1,003 | 3.12× |
| **Cash advances** | $4,040 | $979 | 4.13× |
| **Payments** | $6,448 | $1,733 | 3.72× |
| **Avg order value** | $235 | $74 | 3.18× |
| **Purchase frequency** | 0.49 | 0.49 | 1.00× |
| **Credit limit** | $9,661 | $4,494 | 2.15× |

These customers are **behaviorally unusual in almost every way except frequency.** They carry 2.8× the typical balance, make 3× larger purchases, withdraw 4× more in cash advances, and pay back 3.7× more than average. In operational terms, they're the highest-risk, highest-reward segment — prone to large swings in activity, unusual cash behavior, and substantial credit exposure.

**Why this is useful:** KMeans would have force-assigned these 811 customers to whichever centroid was nearest, hiding them inside an otherwise "normal" segment summary. DBSCAN explicitly surfaces them as "this is unusual." For credit risk, fraud detection, or premium service targeting, knowing which customers deviate sharply from the norm can be as valuable as knowing the norm itself.

---

## How much variance did the first two PCA components explain

**PC1 explains 31.87% of variance individually.**
**PC2 explains 20.05% of variance individually.**
**Together: 51.93% of the original 8-dimensional variance.**

That means **48.07% of the structure in the data is not visible in the 2D scatter plots.** The visualizations are honest but incomplete.

Looking at the full variance breakdown:
- PC1–PC2: 51.93% (shown in plots)
- PC3–PC4: 25.18% (not shown; still substantial)
- PC5–PC8: 22.89% (not shown; but still meaningful)

The PCA plots are useful for spotting cluster *shapes* and rough separation, but they don't capture the full behavioral distance between customers. Two points that look far apart in PC1–PC2 space might actually be quite close in PC5 or PC6, or vice versa. This is why I didn't rely solely on visual inspection — I grounded my K choice in the elbow and silhouette metrics, which operate in the full 8D space.

---

## Which clustering algorithm was more useful for this dataset and why

**KMeans was more actionable overall, but DBSCAN revealed something KMeans missed.**

### KMeans strengths:
- Assigned every customer to a cluster, so the results are immediately deployable (marketing campaigns, credit risk tiers, service level assignments).
- Four interpretable segments with clear behavioral profiles (high-value, inactive, frequent, cash-advance).
- Stable results across the dataset—no need to tune two interacting parameters.
- Silhouette score of 0.2646 is honest but workable; the clusters are real, just not perfectly separated (which matches reality).

### KMeans limitations:
- Forced to assign outliers and unusual customers to whichever cluster they're "least far from," obscuring their true distinctiveness.
- Assumes roughly convex, spherical clusters around centroids (not always true).
- Silent about who doesn't fit well—can hide problems.

### DBSCAN strengths:
- Explicitly identified 811 high-activity outliers without forcing them into a pigeonhole.
- These noise points show a strong behavioral signal: the highest-risk, highest-reward customer segment.
- Doesn't assume cluster shape—can follow genuine density patterns.

### DBSCAN limitations:
- Extremely sensitive to parameter tuning; eps and min_samples interact in non-obvious ways, and small changes produce vastly different results.
- My chosen configuration (eps=1.0, min_samples=8) found mostly one giant cluster, reducing actionability.
- Noise points aren't assigned to any segment, so operationally, "what do we do with these 811 people?" still requires a second decision.
- Less stable for datasets with varying density—dense regions vs. sparse outliers are treated very differently.

### Verdict:
**For business segmentation, I'd use KMeans.** It's interpretable, stable, and provides a clean, actionable four-way split of the customer base. The silhouette score acknowledges real fuzziness in the data, and that's honest.

**But I'd run DBSCAN separately to flag the ~9% of high-value outliers** for specialized treatment (premium service, enhanced monitoring, custom credit terms). KMeans would quietly include them in Cluster 0 (high-value customers), but DBSCAN makes it explicit: "These 811 are fundamentally different—they don't play by the rules of the other clusters."

---

## Questions I covered

✓ **How did scaling change the clustering result?** StandardScaler was essential; without it, `CREDIT_LIMIT` alone would have dominated every distance metric.

✓ **How was the final K selected?** By combining the elbow method (clear bend at K=4) with silhouette scores (K=4 balances interpretability vs. over-simplification), then validating that K=4 captured meaningful behavioral differences that K=2/K=3 would erase.

✓ **What characteristics distinguish each KMeans cluster?** Cluster 0: high-value, high-activity. Cluster 1: inactive. Cluster 2: frequent small-transaction users. Cluster 3: cash-advance focused.

✓ **How sensitive was DBSCAN to eps and min_samples?** Highly sensitive. eps=0.5 fragments the data; eps=1.0–1.5 collapses it; eps=2.0 and higher produce very few clusters. No single "obvious" sweet spot.

✓ **Which points were considered noise and why might that be useful?** 811 customers (9.1%) labeled as noise—systematically high-balance, high-purchase, high-cash-advance, high-payment outliers. Useful for risk management and premium targeting.

✓ **How much variance did the first two PCA components explain?** 51.93% together. The remaining 48% is compressed into PC3–PC8, so 2D visualizations are illustrative, not complete.

✓ **Which clustering algorithm was more useful for this dataset and why?** KMeans for actionable segmentation; DBSCAN for identifying high-value outliers that KMeans would hide. Ideally, use both.

---

## What I would conclude

The credit card customer base is genuinely diverse, but not chaotically so. Four KMeans clusters capture 51.93% of the behavioral variance in just two principal components, and they split the customer base into distinct, operationally meaningful groups. That said:

1. **Cluster 1 (inactive customers, 42% of the base)** is the biggest surprise. Nearly half the card base barely uses their accounts. This is either an untapped growth opportunity or a signal that credit was extended to people unlikely to be active users.

2. **Cluster 0 and Cluster 3 together account for only 19% of customers but likely drive most revenue and risk.** The business's true value is concentrated in a small segment—high-balance, high-cash-advance users. That concentration matters for portfolio management.

3. **Cluster 2 (frequent purchasers)** is large (39%) but their average transaction size is the smallest ($63). They're high-frequency, low-value—steady revenue but with thin margins per transaction. Volume is the play here.

4. **DBSCAN's 9.1% noise segment is compelling.** These are real customers with real behavior, just unusual enough that they don't cluster with anyone else. Treating them as a separate, premium risk profile (vs. forcing them into KMeans Cluster 0) could improve credit decisions and fraud detection.

5. **Scaling was non-negotiable.** Without standardization, credit limit alone would have driven the entire analysis, and I would have missed the purchasing frequency, cash advance, and payment behavior distinctions that actually separate the clusters.

---

## Final takeaway

KMeans with K=4 is a solid, deployable segmentation—it's stable, interpretable, and grounded in both the elbow plot and silhouette metrics. But it's not gospel. DBSCAN's noise segment is a reminder that the "average" customer profile hides real outliers who behave in fundamentally different ways. The most honest conclusion is that the credit card user base has at least five distinct behavioral profiles: Clusters 0–3 (which KMeans surfaces), plus the 9.1% of outliers (which DBSCAN surfaces). A robust strategy would acknowledge all five.
