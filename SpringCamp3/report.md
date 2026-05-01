# Neural Recommender System - Report

## Approach

Given a dataset of implicit user-item interactions, build three models with Matrix Facotrisation, MLP and NeuMF as item recommenders. 
* The task is framed as binary classification:
  * Observed interactions = positive = 1
  * Sampled non-interactions = negative = 0

Models are evaluated uisng Hit@K which is a ranking metric not a classification accuracy.
Following the goal of ranking I tried another model - Bayesian Personalised Ranking whose approach for implicit feedback aligned with the ranking metric.

## Data
- Basic analysis and inferences:
  - 942 users, 1447 items, 55375 interactions
  - $\text{Density} = \frac{\text{Total Interactions}}{\text{UniqueUsers * UniqueItems}} = 4.06$ - Highly sparse
  - All ID's are 0-indexed and continuous so no remapping needed before embedding
  
- Top 10 users only contribute `5.1%` so no apparent long tail
- `439 (30%)` items with `<5` interactions, significant long tail 
     
The item distribution has a significant long tail - their embeddings will be undertrained regardless of model complexity - this is a data coverage problem.

- **Split:** Leave-one-out per user for both test and validation. One interaction per user held out for each, the rest used for training. This ensures every user has at least one interaction in training and gives a cleaner evaluation that aligns with Hit@K. Standard train_test_split would scatter interactions randomly and could leave some users entirely out of evaluation.
Since at minimum user has only 3 interaction even after taking one out for test and validation they are present in train.

**Negative sampling:** For each positive training pair, 3/4 negatives were sampled uniformly from items the user had never interacted with. At 4% density, randomly sampled items are true negatives ~96% of the time - the noise from occasionally sampling an actually-positive item behaves like label noise and can be justified as regularisation. Negatives are only sampled for training as Hit@K evaluation has its own sampling procedure (100 negatives/interaction).

## Models

### Embeddings

An embedding is a parameter matrix $E \in \mathbb{R}^{N \times d}$ where row i is the learned vector for entity i. By end of training, geometrically close embeddings correspond to users/items with similar interaction patterns.

### 1. Matrix Factorization (MF)

MF assumes user-item interactions can be represented as a dot product in a latent space: $\hat{y_ui} = p_u^T q_i$

Each embedding dimension contributes independently - there are no cross-feature interactions or nonlinear dependencies.

**Model considerations:**
- **Similarity metric** - A single dot product means MF assumes the actual interaction data and its latent factors live in a 2D / low rank space. This makes the interaction representation computationally cheap, other kernels like (polynomial,RBF,MLP) would show interactions at a higher rank space but more computation and variance (requiring more data). 
- **Embedding init and normalisation** - bad initialisation can cause score inflation - either ensure proper init or use regularisation (L2/clipping)
- **Bias terms** - Adding bias terms, $b_u$ and $b_i$ would encode global effects (overall popularity,user activity) which can dominate personal preferences. In implicit feedback + negative sampling bias is not preferred at all cause it more so helps in calibration like with actual ratings task with regression than just ranking in our case; we can try a heavily normalised only item bias as it could help with users with very less true feedback
- **Objective:** With binary labeled data BCE loss is favorable but at a higher level we are still using the negative samples which can cause overfitting in regard to that sampling artifact. Another metric is Bayesian Personalised Ranking (BPR) which measures relative order directly which optimises dierectly for ranking based evalutation as in this case
- **Main considations:** 
    - Negative sampling ratio
    - Regularisation on embeddings
    - Embedding dimension / capacity
    - Optimiser 
    - Evaluation

### 2. MLP Recommender

Concatenation gives the MLP access to all user and item features jointly. 

**Model choices:** 
- MLP layer architecture is 64 -> 16 -> 1 : Finally we want a score for each of the user,item pairs we have and compute loss and update our weights, instead of mapping directly to a more strict score like in MF (no hidden layers and hardcoded interaction function) intermediate layers allow for restructring and combination of features progressively before a scalar prediction.
  
- ReLU: Introduces nonlinearity to learn non linear interactions and is applied only in hidden layers so output space remaines unconstrained
- Concatenation instead of dot product: interactions are learned rather than a predefined dot product

* MF is just a very constrained MLP with 
  - No hidden layers
  - Fixed weights - implicitly 1 for matching dimensions
  - Only element-wise interactions
  

### 3. NeuMF

MF is too rigid for complex interactions; MLP generalises but needs more data to justify capacity. NeuMF runs both in parallel with separate embedding tables and learns how to combine outputs:

- MF path: element-wise product, shape (B, mf_dim) - keeps the vector, not reduced to scalar
- MLP path: concat -> layers -> last hidden, shape (B, h_last)
- Predict layer: concat both -> linear -> scalar score

The final layer learns weighting between linear and nonlinear components jointly. Separate embedding tables because MF dot product and MLP concatenation impose different structural constraints on the latent space.


### 4. BPR (Bayesian Personalised Ranking)

BCE with negative sampling treats each (user, item, label) pair independently - the model pushes positives high and negatives low in isolation. BPR optimises directly for ranking: for each user u, a positive item i should score higher than a negative item j:

$\text{BPR} = \sum_{(u,i,j)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})$

Training samples are triplets (u, i, j). Loss pushes the margin $\hat{y}_{ui} - \hat{y}_{uj} > 0$ - which is directly what Hit@K measures. 



## Results

| Model | Test Hit@10 | Val@10 peak | Config |
|-------|-------------|-------------|--------|
| BPR   | 0.7611      | 0.7431      | emb=32, n_neg=4, lr=1e-3, ep=6 |
| MLP   | 0.7580      | 0.7834      | emb=32, [64,16], lr=1e-2, ep=14 |
| NeuMF | 0.7155      | 0.6868      | mf/mlp_dim=24, [32,16], lr=5e-3 |
| MF    | 0.7144      | 0.7282      | emb=16, lr=1e-3, ep=10 |

## Analysis

**MLP > MF** - MLP learns a nonlinear interaction function instead of being constrained to a dot product. It generalises better: val@10 reaches 0.783 and holds, whereas MF reaches 0.728 and falls.

**NeuMF not beating MLP** - NeuMF has 4 embedding tables vs 2 in MLP. On 55k interactions, val@10 flatlines from epoch 1 to epoch 6 (0.687 both) - I tried out multiple configs but it plateaus around 0.68-0.7. This is most likely due to the high capacity NeuMF inherently has which doesnt match with this small dataset and it starts memorising too quickly.

* BPR with n_neg=4 gives 4 pairwise comparisons per positive per step -> better gradient signal, and the ranking margin objective is directly what Hit@K measures.

A Hit@10 difference of 0.02 corresponds to ~19 users out of 942 so we cant draw much conclusive results from these runs.

**Used:**
- validation-guided epoch selection (stopped when val@10 stopped improving rather than running fixed epochs)
- varied negative sampling ratio  emb_dim search across MF (16 vs 32 - smaller won on this dataset)

