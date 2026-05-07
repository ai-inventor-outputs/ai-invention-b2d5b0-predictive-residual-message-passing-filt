# PRMP Arch Spec

## Summary

Complete architectural specification for Predictive Residual Message Passing (PRMP), a novel message passing mechanism for heterogeneous GNNs on relational databases. Covers PredNet-inspired predict-subtract-propagate equations, Kalman filter analogy, full mathematical formulation with gradient flow analysis, PyG MessagePassing/HeteroConv integration strategy, design decisions (MLP sizing, normalization, initialization, gradient detachment), six ablation variants, and positioning relative to RelGNN and Griffin. Synthesized from 14 primary sources spanning predictive coding, signal processing, GNN frameworks, and relational deep learning benchmarks.

## Research Findings

## Complete Architectural Specification for Predictive Residual Message Passing (PRMP)

### 1. Introduction: The Aggregation Bottleneck at FK Joins

In relational deep learning, heterogeneous temporal graphs are constructed from relational database tables where each table becomes a node type and primary-foreign key (PK-FK) relationships define edges [1, 2]. The standard approach uses heterogeneous GraphSAGE with sum-based neighbor aggregation across FK joins [1]. However, at high-cardinality FK joins (e.g., a popular product with thousands of reviews), the aggregation must compress many child node features into a single parent representation. Much of this information is redundant — predictable from the parent's own features — yet standard aggregation treats all child contributions equally, wasting representational capacity on unsurprising information.

PRMP addresses this by introducing a predict-subtract-propagate mechanism inspired by predictive coding [3, 4] and Kalman filtering [5], where a learned prediction of each child's features is subtracted before aggregation, so only the surprising residual information is passed upward.

### 2. Background

#### 2.1 PredNet: Predict-Subtract-Propagate (Lotter et al., ICLR 2017)

PredNet [3] implements predictive coding in a deep convolutional recurrent neural network for video prediction. Each layer l contains four component types: representation neurons R_l, predictions Â_l, targets A_l, and error neurons E_l. The update equations are [3]:

1. **Representation Update:** R_l^t = ConvLSTM(E_l^{t-1}, R_l^{t-1}, Upsample(R_{l+1}^t))
2. **Prediction:** Â_l^t = ReLU(Conv(R_l^t))
3. **Error (Residual):** E_l^t = [ReLU(A_l^t - Â_l^t) ; ReLU(Â_l^t - A_l^t)] — rectified positive and negative prediction errors concatenated
4. **Target Propagation:** A_{l+1}^t = Conv(MaxPool(E_l^t))
5. **Training Loss:** L = Σ_t Σ_l (λ_t · λ_l / n_l) · Σ_i E_{l,i}^t

**Key design choices relevant to PRMP:** PredNet's split-ReLU separates positive/negative errors, but PRMP uses simple subtraction since residuals feed into learned aggregation. R_l and E_l are initialized to zero — analogous to PRMP's zero-initialization strategy. PredNet operates in a spatial hierarchy (CNN layers); PRMP adapts this to a relational hierarchy (FK edge types) [3].

#### 2.2 Salvatori et al.: Predictive Coding on Graphs (NeurIPS 2022)

Salvatori et al. [4] showed how predictive coding can perform inference and learning on arbitrary graph topologies (PC graphs). Their formulation uses [6, 7]:

- **Energy Function:** F = ½ Σ_l (ε_l)² where ε_l = a_l - μ_l
- **Inference Update:** Δa_l = -γ(ε_l - (w_l)^T(ε_{l+1} ⊙ f'(w_l·a_l)))
- **Weight Update:** Δw_l = α · ε_{l+1} ⊙ f'(w_l·â_l) · (â_l)^T

**Critical distinction from PRMP:** Salvatori uses PC as a *learning algorithm* (alternative to backprop, requiring 10-25 iterative inference steps per training iteration [7]). PRMP instead uses predict-subtract as an *architectural design* within standard backprop, avoiding convergence and scalability concerns [4, 7].

#### 2.3 Kalman Filter Analogy

The Kalman filter [5] provides the signal-processing foundation for PRMP:

| Kalman Filter | PRMP Analog | Interpretation |
|---|---|---|
| F·x̂ (state prediction) | f_e(h_parent) | Prediction MLP maps parent embedding to predicted child |
| z_k (measurement) | h_child | Actual child node features |
| ỹ_k (innovation) | r_child = h_child - f_e(h_parent) | Residual: what parent cannot predict about child |
| K_k (Kalman gain) | Learnable gating/attention | Controls residual trust |
| State update | h_parent_new = h_parent + AGGR(g(r)) | Parent updated with aggregated weighted residuals |

### 3. PRMP Mathematical Formulation

#### 3.1 Per-Edge-Type Prediction MLP
For each FK edge type e = (parent_type, fk_rel, child_type):
```
f_e(h_parent) = W_2 · ReLU(W_1 · h_parent + b_1) + b_2
```
where d_hidden = min(d_parent, d_child).

#### 3.2 Residual Computation
```
r_child = LayerNorm(h_child - f_e(h_parent.detach()))
```

#### 3.3 Residual Aggregation (default: mean)
```
m_parent = (1/|N(parent)|) · Σ_{child ∈ N(parent)} r_child
```

#### 3.4 Parent Update (GraphSAGE-style)
```
h_parent^{(l+1)} = ReLU(W_update · [h_parent^{(l)} || m_parent] + b_update)
```

#### 3.5 Multi-Edge-Type (HeteroConv default)
```
h_parent^{(l+1)} = Σ_{e ∈ E(parent)} m_parent^{(e)}
```

### 4. Gradient Flow Analysis

The key gradient through the prediction MLP: ∂r_child/∂f_e = -I (negative identity). This means the prediction MLP is trained to predict whatever portion of child features is NOT useful for the downstream task — making prediction task-adaptive [3, 5]. With detach (recommended), parent embeddings receive no gradient through the prediction path, preventing competition between prediction and task update gradients.

**Collapse protection:** Zero initialization ensures r_child ≈ h_child at training start (equivalent to standard aggregation), and end-to-end training prevents collapse since zero residuals = zero task gradient = loss increases = f_e adjusts.

### 5. Design Decisions

Key defaults: 2-layer MLP with zero-initialized final layer, LayerNorm on residuals, gradient detach on parent input, separate MLP per edge type, mean aggregation.

### 6. Six Ablation Variants

- **A: Random predictions** (frozen MLP control)
- **B: No-subtraction** (concatenate instead — skip connection control)
- **C: Selective PRMP** (high-cardinality joins only)
- **D: Linear prediction** (single linear layer capacity control)
- **E: Detach vs no-detach** (gradient competition test)
- **F: Normalization variants** (none, LayerNorm, gated)

### 7. PyG Implementation Strategy

Custom `PRMPConv(MessagePassing)` subclass overriding `message()` to compute residuals using `x_j` (child/source) and `x_i` (parent/destination) features [8]. Wrapped in `HeteroConv` with per-edge-type instances [9]. Fully compatible with RelBench's temporal neighbor sampling and PyTorch Frame feature encoding [1, 14].

### 8. Positioning vs Prior Work

- **RelGNN** [10]: Addresses routing inefficiency (atomic routes). PRMP addresses information redundancy (residual filtering). Orthogonal and composable.
- **Griffin** [11]: Cross-attention for within-node cell-level features. PRMP targets between-node FK message passing. Complementary bottlenecks.
- **PC-GNNs** [12]: Use PC as learning algorithm (iterative inference). PRMP uses prediction-subtraction architecturally within standard backprop.

## Sources

[1] [RelBench: A Benchmark for Deep Learning on Relational Databases](https://arxiv.org/html/2407.20060v1) — Defines the RelBench benchmark with 7 datasets, 30 tasks, heterogeneous temporal graph construction from relational tables, and the GraphSAGE baseline architecture with sum aggregation, 128 hidden dim, 2 layers.

[2] [Relational Deep Learning: Graph Representation Learning on Relational Databases](https://relbench.stanford.edu/paper.pdf) — Original relational deep learning paper describing how relational tables map to heterogeneous graphs with PK-FK edges, and how message passing GNNs can learn across multiple tables.

[3] [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning (PredNet)](https://arxiv.org/abs/1605.08104) — Source of the predict-subtract-propagate mechanism. Provides the four core equations: R_l update via ConvLSTM, prediction via Conv(R_l), error via split-ReLU subtraction, and target propagation. Key design template for PRMP.

[4] [Learning on Arbitrary Graph Topologies via Predictive Coding (Salvatori et al., NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f9f54762cbb4fe4dbffdd4f792c31221-Abstract-Conference.html) — Extends predictive coding to arbitrary graph topologies (PC graphs). Uses PC as a learning algorithm with iterative inference. Key contrast with PRMP which uses prediction-subtraction architecturally within standard backprop.

[5] [Kalman Filter - Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter) — Provides the standard Kalman filter equations: state prediction, innovation/residual computation, Kalman gain, and state update. Foundation for the PRMP mathematical analogy.

[6] [Predictive Coding Networks and Inference Learning: Tutorial and Survey](https://arxiv.org/html/2407.04117v1) — Comprehensive tutorial on PCN equations: energy F = 1/2 sum (epsilon_l)^2, inference update for value nodes, weight update rule, and distinction between PC as learning algorithm vs architectural principle.

[7] [Predictive Coding: A Brief Introduction and Review for Machine Learning Researchers](https://neuralnetnick.com/2022/12/28/predictive-coding-a-brief-introduction-and-review-for-machine-learning-researchers/) — Clear exposition of PCN energy function, value node update rule with inference rate gamma, and Hebbian-like weight update. Notes PC approximates implicit SGD.

[8] [Creating Message Passing Networks - PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) — Complete PyG MessagePassing API: propagate() -> message() -> aggregate() -> update() lifecycle, _i/_j suffix convention, bipartite x=(x_src, x_dst) support, code examples.

[9] [Heterogeneous Graph Learning - PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html) — HeteroConv wrapper taking Dict[edge_type -> MessagePassing], aggr parameter for multi-edge-type aggregation, HeteroData format with x_dict/edge_index_dict.

[10] [RelGNN: Composite Message Passing for Relational Deep Learning](https://arxiv.org/html/2502.06784v1) — Introduces atomic routes and composite message passing. FUSE: W1*h_mid + W2*h_src, attention aggregation with per-route weight matrices. Surpasses baselines on 27/30 RelBench tasks. Addresses routing inefficiency (orthogonal to PRMP).

[11] [Griffin: Towards a Graph-Centric Relational Database Foundation Model](https://arxiv.org/html/2505.05568v1) — Cross-attention for within-node feature aggregation with task-conditioned queries. Hierarchical aggregation: within-relation Mean, cross-relation Max with relation embeddings. Targets cell-level features (complementary to PRMP).

[12] [Robust Graph Representation Learning via Predictive Coding](https://arxiv.org/abs/2212.04656) — Applies PC message-passing rule to GNN architectures, showing comparable performance, better calibration, and adversarial robustness. Uses PC as learning algorithm, unlike PRMP's architectural approach.

[13] [GGNNs: Generalizing GNNs using Residual Connections and Weighted Message Passing](https://arxiv.org/html/2311.15448) — Introduces inter-layer residual connections and weighted message passing for GNNs. Different from PRMP which computes residuals within the message function between predicted and actual child features.

[14] [RelBench GitHub Repository](https://github.com/snap-stanford/relbench) — Source code for RelBench including gnn_entity.py example, HeteroGraphSAGE model, temporal neighbor sampling, and PyTorch Frame integration.

## Follow-up Questions

- How does the prediction accuracy of f_e vary across different FK edge types and RelBench datasets, and does the R-squared between parent-predicted and actual child features correlate with PRMP's improvement over baseline?
- Can PRMP be composed with RelGNN's composite message passing by applying residual computation to the fused features before attention aggregation, and does this yield additive improvements?
- What is the computational overhead of PRMP's per-edge-type prediction MLPs relative to standard aggregation, and does the overhead scale linearly with the number of FK edge types in the schema?

---
*Generated by AI Inventor Pipeline*
