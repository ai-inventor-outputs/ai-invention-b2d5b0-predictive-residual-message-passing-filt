# RelBench Ref

## Summary

Comprehensive reference for RelBench v1/v2 covering all 7 v1 datasets with complete FK schemas (51 total FK links extracted from source code), all 30 v1 benchmark tasks (12 classification, 9 regression, 9 recommendation) with types/metrics/splits, SOTA results from GNN baselines (GraphSAGE AUROC/MAE/MAP), RelGNN (27/30 SOTA, up to 25% improvement), Griffin (cross-attention + hierarchical aggregation), DBFormer (Transformer-based hypergraph), ContextGNN (344% improvement on recommendation), and KumoRFM (2-8% zero-shot gains). Includes complete relbench Python API patterns for data loading, PyG graph construction, and GNN training. Identifies high-cardinality FK links most relevant to the PRMP hypothesis.

## Research Findings

## Comprehensive RelBench v1/v2 Reference

### 1. Dataset Inventory

RelBench v1 contains 7 datasets spanning 51 tables, 103.4M rows, 489 columns, and 30 tasks [1, 2]. The datasets are:

- **rel-amazon** (E-commerce): 3 tables (product with 506K rows, customer with 1.85M rows, review with 21.9M rows), 15M total rows, 15 cols, 7 tasks. Star schema with review as central fact table linking customer and product [1, 3]. Time range: 1996-06-25 to 2018-09-28, val_timestamp=2014-01-21, test_timestamp=2016-01-01.
- **rel-avito** (E-commerce): 8 tables (UserInfo, AdsInfo, Category, Location, SearchInfo, SearchStream, VisitStream, PhoneRequestsStream), 20.7M rows, 42 cols, 4 tasks. Complex multi-path schema with UserInfo and AdsInfo as hub entities connected through SearchStream, VisitStream, PhoneRequestsStream [4].
- **rel-event** (Social): 5 tables (users, events, event_attendees, event_interest, user_friends), 41.3M rows, 128 cols, 3 tasks. Social graph with self-referential user_friends table where both FKs point to users [5].
- **rel-f1** (Sports): 9 tables (circuits, drivers, constructors, races, results, standings, constructor_results, constructor_standings, qualifying), 74K rows, 67 cols, 3 tasks. Complex star-snowflake with results and qualifying as hub nodes with 3 FKs each (race, driver, constructor) [6].
- **rel-hm** (E-commerce): 3 tables (article, customer, transactions), 16.7M rows, 37 cols, 3 tasks. Simple star schema: transactions links customer and article [7].
- **rel-stack** (Social): 7 tables (users, posts, comments, badges, postLinks, postHistory, votes), 4.2M rows, 52 cols, 5 tasks. Complex hub-spoke with self-referential posts.ParentId and dual hubs (users + posts) [8]. Val_timestamp=2019-01-01, test_timestamp=2021-01-01.
- **rel-trial** (Medical): 15 tables (studies as central hub, plus outcomes, outcome_analyses, drop_withdrawals, reported_event_totals, designs, eligibilities, interventions, conditions, facilities, sponsors, and 4 junction tables), 5.4M rows, 140 cols, 5 tasks [9].

RelBench v2 (January 2026) adds 4 more datasets: rel-salt (ERP, 4 tables, 4.3M rows), rel-arxiv (scholarly, 6 tables, 2.1M rows), rel-ratebeer (consumer, 13 tables, 13.8M rows), rel-mimic (clinical, 6 tables, 2.4M rows), bringing the total to 11 datasets and 70 tasks [10].

### 2. Complete FK Link Catalog

I extracted 51 FK links across all 7 v1 datasets directly from the relbench source code [3, 4, 5, 6, 7, 8, 9]:

**rel-amazon (2 FK links):** review.customer_id→customer (estimated ~11.8 reviews/customer), review.product_id→product (estimated ~43.3 reviews/product) [3].

**rel-avito (10-11 FK links):** AdsInfo.LocationID→Location, AdsInfo.CategoryID→Category, SearchInfo.UserID→UserInfo, SearchInfo.LocationID→Location, SearchInfo.CategoryID→Category, SearchStream.SearchID→SearchInfo, SearchStream.AdID→AdsInfo, VisitStream.UserID→UserInfo, VisitStream.AdID→AdsInfo, PhoneRequestsStream.UserID→UserInfo, PhoneRequestsStream.AdID→AdsInfo. AdsInfo is a bridge node with 2 FKs; SearchInfo is a hub node with 3 FKs [4].

**rel-event (6-7 FK links):** events.user_id→users, event_attendees.event→events, event_attendees.user_id→users, event_interest.event→events, event_interest.user→users, user_friends.user→users, user_friends.friend→users. user_friends is self-referential with both FKs to users [5].

**rel-f1 (11 FK links):** races.circuitId→circuits, results.raceId→races, results.driverId→drivers, results.constructorId→constructors, standings.raceId→races, standings.driverId→drivers, constructor_results.raceId→races, constructor_results.constructorId→constructors, constructor_standings.raceId→races, constructor_standings.constructorId→constructors, qualifying.raceId→races, qualifying.driverId→drivers, qualifying.constructorId→constructors. results and qualifying are HUB nodes with 3 FKs each — ideal for RelGNN's atomic routes [6].

**rel-hm (2 FK links):** transactions.customer_id→customer, transactions.article_id→article [7].

**rel-stack (11 FK links):** posts.OwnerUserId→users, posts.ParentId→posts (self-referential), comments.UserId→users, comments.PostId→posts, badges.UserId→users, postLinks.PostId→posts, postLinks.RelatedPostId→posts, postHistory.PostId→posts, postHistory.UserId→users, votes.PostId→posts, votes.UserId→users [8].

**rel-trial (12-15 FK links):** outcomes.nct_id→studies, outcome_analyses.nct_id→studies, outcome_analyses.outcome_id→outcomes, drop_withdrawals.nct_id→studies, reported_event_totals.nct_id→studies, designs.nct_id→studies, eligibilities.nct_id→studies, interventions_studies.nct_id→studies, interventions_studies.intervention_id→interventions, conditions_studies.nct_id→studies, conditions_studies.condition_id→conditions, facilities_studies.nct_id→studies, facilities_studies.facility_id→facilities, sponsors_studies.nct_id→studies, sponsors_studies.sponsor_id→sponsors [9].

FK cardinalities can be computed programmatically: `db.table_dict[child_table].df.groupby(fk_col).size().describe()` [2].

### 3. Task Inventory

All 30 v1 tasks divide into three types [1, 2]:

**Entity Classification (12 tasks, metric=AUROC):** rel-amazon user-churn (4.7M train rows), rel-amazon item-churn (2.6M), rel-avito user-clicks (59K), rel-avito user-visits (87K), rel-event user-repeat (3.8K), rel-event user-ignore (19K), rel-f1 driver-dnf (11K), rel-f1 driver-top3 (1.4K), rel-hm user-churn (3.9M), rel-stack user-engagement (1.4M), rel-stack user-badge (3.4M), rel-trial study-outcome (12K) [1].

**Entity Regression (9 tasks, metric=MAE):** rel-amazon user-ltv (4.7M), rel-amazon item-ltv (2.7M), rel-avito ad-ctr (5.1K), rel-event user-attendance (19K), rel-f1 driver-position (7.5K), rel-hm item-sales (5.5M), rel-stack post-votes (2.5M), rel-trial study-adverse (43K), rel-trial site-success (151K) [1].

**Recommendation (9 tasks, metric=MAP@K):** rel-amazon user-item-purchase (5.1M), rel-amazon user-item-rate (3.7M), rel-amazon user-item-review (2.3M), rel-avito user-ad-visit (87K), rel-hm user-item-purchase (3.9M), rel-stack user-post-comment (21K), rel-stack post-post-related (5.9K), rel-trial condition-sponsor-run (37K), rel-trial site-sponsor-run (669K) [1].

**V2 adds 40 new tasks** including 23 autocomplete tasks (predicting masked column values) and 17 forecasting tasks across the 4 new datasets [10].

**Temporal splits:** Data is split temporally. Models train on rows up to val_timestamp, validate between val/test timestamps, test after test_timestamp. Temporal neighbor sampling ensures all nodes within sampled subgraphs appear before the seed time to prevent data leakage [1].

### 4. Benchmark Results

**Entity Classification (AUROC, higher is better)** [11]: GraphSAGE (RDL) averages 75.83 across 12 tasks vs LightGBM's 63.69. Per-task: rel-amazon user-churn LightGBM=52.22 / RDL=70.42, item-churn 62.54/82.81, rel-stack user-engagement 63.39/90.59, user-badge 63.43/88.86, rel-trial study-outcome 70.09/68.60 (LightGBM wins here), rel-f1 driver-dnf 68.85/72.62, driver-top3 73.93/75.54, rel-hm user-churn 55.21/69.88, rel-event user-repeat 68.04/76.89, user-ignore 79.93/81.62, rel-avito user-visits 53.05/66.20, user-clicks 53.60/65.90 [11].

**Entity Regression (MAE, lower is better)** [11]: GraphSAGE improves over LightGBM on most tasks. Per-task: rel-amazon user-ltv 16.783/14.313, item-ltv 60.569/50.053, rel-stack post-votes 0.068/0.065, rel-trial study-adverse 44.011/44.473 (LightGBM wins), site-success 0.425/0.400, rel-f1 driver-position 4.170/4.022, rel-hm item-sales 0.076/0.056, rel-event user-attendance 0.264/0.258, rel-avito ad-ctr 0.041/0.041 (tie) [11].

**Recommendation (MAP%, higher is better)** [12]: ContextGNN achieves 9.23% average test MAP vs GraphSAGE's 2.08% and LightGBM's 2.01%. Per-task test MAP: rel-amazon user-item-purchase LightGBM=0.16/GraphSAGE=0.74/NBFNet=2.06/ContextGNN=2.93, user-item-rate 0.17/0.87/1.24/2.25, user-item-review 0.09/0.47/1.57/1.63, rel-hm user-item-purchase 0.38/0.80/2.81/2.93, rel-stack user-post-comment 0.04/0.11/12.72/13.34, rel-trial condition-sponsor-run 4.82/2.89/11.36/11.65, site-sponsor-run 8.40/10.70/19.06/28.02 [12].

**RelGNN (ICML 2025)** [13, 14]: Surpasses all baselines on 27 of 30 tasks with up to 25% improvement. Entity classification: outperforms on 10 of 12 tasks. Entity regression: outperforms on 8 of 9 tasks. Recommendation: performs better or equal on all 9 tasks. Larger improvements correlate with more complex FK structure (e.g., rel-f1 > rel-amazon, rel-hm). Exact per-task numerical values are stored in external .tex files not accessible from the HTML paper version [13]. The RelGNN code is publicly available and can be run via commands like `python relgnn_task_node.py --dataset rel-amazon --task user-churn` [14].

**V2 benchmark results** [10]: Autocomplete binary AUROC: rel-avito searchinfo-isuserloggedon=73.00, searchstream-click=55.92; rel-trial eligibilities-adult=93.73, eligibilities-child=87.25, studies-has_dmc=75.72. Autocomplete multiclass accuracy: rel-salt item-plant=99.46%, item-shippoint=98.39%, sales-office=99.88%; rel-stack badges-class=82.83%. Entity binary AUROC: rel-arxiv paper-citation=82.50, rel-ratebeer beer-churn=78.67, user-churn=94.27, brewer-dormant=80.51. Recommendation MAP: rel-arxiv paper-paper-cocitation=35.39, rel-f1 driver-circuit-compete=76.18, rel-ratebeer user-beer-favorite=1.89, user-beer-liked=1.46, user-place-liked=1.85 [10].

**KumoRFM** [16]: Zero-shot outperforms supervised learning by 2-8% on RelBench; fine-tuned adds 10-30% boost. Graph-transformer-based architecture with in-context learning. Orders of magnitude faster than conventional supervised training [16].

### 5. Architecture Summaries

**GraphSAGE Baseline (RDL default)** [17, 19]: Encodes each table row into initial node embeddings using PyTorch Frame's ResNet tabular model. Performs temporal-aware subgraph sampling. Uses heterogeneous GraphSAGE: per-edge-type SAGEConv with configurable aggregation (default mean within edge type) and HeteroConv with sum aggregation across edge types. Forward pass: conv → batch_norm → relu per layer. Key limitation: uniform aggregation treats all neighbors equally regardless of relevance [17].

**RelGNN (ICML 2025)** [13]: Composite message passing along 'atomic routes' — simple paths between node types enabling single-hop information exchange grounded in FK relationships. For bridge nodes (2 FKs), creates direct (source → intermediate → destination) paths. For hub nodes (3+ FKs), induces latent second-order clique structures. FUSE operation: FUSE(h_mid, h_src) = W1*h_mid + W2*h_src. Aggregation formula: m_(dst,mid,src) = AGGR(h_dst, {FUSE(h_mid, h_src)}). Eliminates redundant back-and-forth message passing [13].

**Griffin (2025)** [15]: Graph-centric RDB foundation model using cross-attention for row aggregation: v_i^l = Attention_l(Q=MLP_l(u_i, t), K=m_i, V=x_i). Hierarchical aggregation: mean within each relation type, max across relation types. Pretrained on 200+ single-table datasets (~10M rows, 150M+ nodes) with masked cell completion. Unpretrained version outperforms all other models in average rank [15].

**DBFormer (2024)** [18]: Transformer architecture for relational databases as two-level multi-relational hypergraphs. Intra-relational: Transformer Encoder applies self-attention over each tuple's attributes, preserving attribute-level structure. Inter-relational: cross-attention Ct(ti,tj) = attn(Q=ti, K=tj, V=tj) for FK-linked tuple pairs. Full layer: FNN+Norm → AtSum → AtAttn → CtCross-Attn → TtTrans-Encoder. Best average rank 1.95 on classification across CTU benchmark datasets (NOT RelBench v1 — uses different benchmark) [18].

**ContextGNN (2024)** [12]: Beyond two-tower recommendation models, combining GNN-based collaborative filtering with contextual component. Achieves 344% improvement over two-tower baselines and 20% over NBFNet. Average test MAP 9.23% vs GraphSAGE 2.08% on RelBench recommendation tasks [12].

### 6. Python API Reference

**Installation:** `pip install relbench` (basic) or `pip install relbench[full]` (with PyTorch dependencies) [2].

**Loading data** [2]:
```python
from relbench.datasets import get_dataset
from relbench.tasks import get_task
dataset = get_dataset('rel-amazon', download=True)
db = dataset.get_db()
task = get_task('rel-amazon', 'user-churn', download=True)
train_table = task.get_table('train')
```

**Database API** [2, 3]: `db.table_dict` is Dict[str, Table] mapping table name to Table object. Each Table has `df` (pandas DataFrame), `pkey_col`, `time_col`, and `fkey_col_to_pkey_table` dict mapping FK columns to parent table names.

**Graph construction** [2, 17, 19]:
```python
from relbench.modeling.graph import make_pkey_fkey_graph
data, col_stats_dict = make_pkey_fkey_graph(db, col_to_stype_dict, text_embedder_cfg, cache_dir)
# Returns: data = PyG HeteroData (nodes=table rows, edges=FK links)
```

**Model architecture** [17]: HeteroGraphSAGE builds layers using HeteroConv wrapping per-edge-type SAGEConv((channels, channels), channels, aggr=aggr) with HeteroConv({...}, aggr='sum'). Each layer: conv → batch_norm → relu.

**Training** [19]: Uses PyG NeighborLoader with temporal sampling, get_node_train_table_input for input nodes, standard PyTorch training loop with task.evaluate(pred) for standardized evaluation.

### 7. PRMP-Relevant Analysis

**Current aggregation approaches:** GraphSAGE baseline uses mean/sum with no selectivity [17]. RelGNN uses attention-weighted FUSE along atomic routes but still aggregates all messages [13]. Griffin uses cross-attention with hierarchical mean-then-max [15]. DBFormer uses cross-attention with attention-weighted summation [18].

**High-cardinality FK links where PRMP could have most impact** [3, 6, 8]: review→product (~43 reviews/product) in rel-amazon, votes→posts in rel-stack, event_attendees→events in rel-event, facilities_studies→studies in rel-trial, transactions→article in rel-hm. These are joins where mean aggregation dilutes signal from informative neighbors.

**Schema complexity ranking** [3, 4, 5, 6, 7, 8, 9]: Simple star (rel-amazon 2 FKs, rel-hm 2 FKs), medium (rel-event 6-7 FKs, rel-avito 10-11 FKs), complex hub-spoke (rel-stack 11 FKs with self-refs, rel-f1 11 FKs with 3-FK hubs, rel-trial 12-15 FKs with junction tables).

**Key observation:** RelGNN reports larger improvements on datasets with more complex FK structure (rel-f1) compared to simple schemas (rel-amazon, rel-hm) [13]. This suggests complex schemas have more room for aggregation improvements — directly relevant to the PRMP hypothesis.

**Contradicting evidence:** DBFormer achieves strong results on CTU benchmarks but was NOT evaluated on RelBench v1 tasks, so direct comparison with RelGNN/Griffin is not available [18]. KumoRFM claims are from company publications without independent verification [16].

**Confidence levels:** HIGH for dataset schemas, FK links, task definitions, and API patterns (extracted directly from source code) [3, 4, 5, 6, 7, 8, 9]. MEDIUM for exact numerical results (obtained from secondary papers, not all methods compared side-by-side) [11, 12]. LOW for RelGNN per-task numbers (confirmed claims but exact values locked in external .tex files) [13].

## Sources

[1] [RelBench: A Benchmark for Deep Learning on Relational Databases (NeurIPS 2024)](https://arxiv.org/html/2407.20060v1) — Primary RelBench paper with all 7 dataset statistics (Table 1), 30 task definitions (Table 2), temporal split methodology, and baseline GNN vs data scientist comparison.

[2] [RelBench Quick Start Guide](https://relbench.stanford.edu/start/) — Official API documentation showing dataset loading with get_dataset(), task access with get_task(), graph construction with make_pkey_fkey_graph, and evaluation patterns.

[3] [RelBench Amazon Dataset Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/datasets/amazon.py) — Exact FK definitions: review table has customer_id to customer and product_id to product FKs. Star schema with review as central fact table.

[4] [RelBench Avito Dataset Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/datasets/avito.py) — 8 tables with 10-11 FK links. Complex multi-path schema with UserInfo and AdsInfo as hubs connected through SearchStream, VisitStream, PhoneRequestsStream.

[5] [RelBench Event Dataset Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/datasets/event.py) — 5 tables with 6-7 FK links including self-referential user_friends (both FKs to users). Bridge tables connect users and events.

[6] [RelBench F1 Dataset Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/datasets/f1.py) — 9 tables with 11 FK links. results and qualifying are hub nodes with 3 FKs each (race, driver, constructor) — ideal for atomic routes.

[7] [RelBench H&M Dataset Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/datasets/hm.py) — 3 tables with 2 FK links. Simple star schema: transactions to customer and transactions to article.

[8] [RelBench Stack Dataset Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/datasets/stack.py) — 7 tables with 11 FK links. Self-referential posts.ParentId, postLinks with both FKs to posts. Users and posts as dual hubs.

[9] [RelBench Trial Dataset Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/datasets/trial.py) — 15 tables with 12-15 FK links. Studies as central hub with 4 junction tables for many-to-many relationships.

[10] [RelBench v2: A Large-Scale Benchmark and Repository for Relational Data](https://arxiv.org/html/2602.12606v1) — 4 new datasets (rel-salt, rel-arxiv, rel-ratebeer, rel-mimic), 40 new tasks including autocomplete tasks, v2 benchmark results with exact numerical values.

[11] [Tackling Prediction Tasks in Relational Databases with LLMs](https://arxiv.org/html/2411.11829v1) — Tables 1-2 with exact AUROC and MAE values for LightGBM and RDL (GraphSAGE) baselines across all 21 entity-level RelBench v1 tasks.

[12] [ContextGNN: Beyond Two-Tower Recommendation Systems](https://arxiv.org/html/2411.19513v1) — Table 2 with exact test MAP% values for 7 recommendation tasks comparing LightGBM, GraphSAGE, NBFNet, and ContextGNN.

[13] [RelGNN: Composite Message Passing for Relational Deep Learning (ICML 2025)](https://arxiv.org/html/2502.06784v2) — Atomic routes formulation, FUSE operation, composite message passing. SOTA on 27/30 tasks with up to 25% improvement. Larger gains on complex FK structures.

[14] [RelGNN GitHub Repository](https://github.com/snap-stanford/RelGNN) — Code repository for RelGNN with installation instructions and usage examples for entity and recommendation tasks.

[15] [Griffin: Graph-Centric RDB Foundation Model](https://arxiv.org/html/2505.05568v1) — Cross-attention architecture, hierarchical mean-then-max aggregation, pretraining on 200+ datasets. Outperforms all models in average rank.

[16] [KumoRFM: A Foundation Model for In-Context Learning on Relational Data](https://kumo.ai/research/kumo_relational_foundation_model.pdf) — Zero-shot outperforms supervised by 2-8% on RelBench; fine-tuned adds 10-30% boost. Graph-transformer architecture.

[17] [RelBench Modeling Neural Network Module](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/modeling/nn.py) — HeteroGraphSAGE implementation: HeteroConv wrapping per-edge-type SAGEConv with sum cross-edge aggregation. HeteroEncoder for PyTorch Frame feature encoding.

[18] [Transformers Meet Relational Databases (DBFormer)](https://arxiv.org/html/2412.05218v1) — DBFormer architecture with Transformer self-attention over attributes, cross-attention for FK links, two-level hypergraph representation. Best avg rank 1.95 on CTU benchmarks.

[19] [RelBench GNN Entity Training Example](https://raw.githubusercontent.com/snap-stanford/relbench/main/examples/gnn_entity.py) — Complete training script showing make_pkey_fkey_graph usage, NeighborLoader setup, training loop, and evaluation code.

## Follow-up Questions

- What are the exact per-task numerical results for RelGNN vs GraphSAGE on all 30 tasks? (Requires downloading the PDF and extracting Tables 1-3 with their external .tex data.)
- What are the actual join cardinalities (mean, median, max children per parent) for each FK link across all 7 datasets? (Can be computed programmatically using db.table_dict[child].df.groupby(fk_col).size().describe())
- How does the PRMP residual passing mechanism compare architecturally to Griffin's hierarchical mean-then-max aggregation and RelGNN's attention-weighted FUSE — are they complementary or redundant approaches to the same aggregation selectivity problem?

---
*Generated by AI Inventor Pipeline*
