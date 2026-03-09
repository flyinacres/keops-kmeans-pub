# KeOps K-Means Clustering

An investigation into GPU-accelerated k-means clustering using [KeOps](https://www.kernel-operations.io/keops/),
with comparisons against scikit-learn and pure PyTorch implementations.

Designed to run on Kaggle with GPU acceleration enabled.

## What This Is

K-means clustering requires computing the distance from every point to every centroid at
each iteration. With N points, K centroids, and D dimensions, the naive approach allocates
an `(N, K, D)` intermediate tensor that grows quickly -- 500k points at 50 dimensions and
K=12 requires about 1.2 GB per iteration in float32, just for that one intermediate.

KeOps solves this by representing the computation symbolically as a `LazyTensor` and
compiling a fused CUDA kernel that streams through the data in registers, never writing
the intermediate to GPU memory. The result scales to datasets that would otherwise exhaust
VRAM.

This notebook builds three implementations of Lloyd's algorithm to make that difference
concrete, then benchmarks them across six datasets.

## Implementations

| Name | Description |
|------|-------------|
| `kmeans_keops` | KeOps `LazyTensor` implementation. Memory-efficient via kernel fusion. |
| `kmeans_tensor` | Plain PyTorch. Materializes the full `(N, K, D)` distance matrix. Illustrates the memory cost KeOps avoids. |
| `kmeans_sklearn` | scikit-learn `KMeans` wrapper. Used as a correctness and timing baseline. |
| `kmeans_totally_tensor` | Experimental: batches all `n_init` restarts into a single tensor operation to test whether parallelism can substitute for kernel fusion. |

## Centroid Initialization Strategies

Four strategies are implemented and compared, all with a common interface `(x, K, n_init) -> (n_init, K, D)`:

- **random** -- sample K distinct points uniformly at random
- **kmeans++** -- distance-weighted sequential selection; reduces sensitivity to bad starts
- **mean_std** -- place initial centroids using per-dimension statistics
- **kppp** (k+++) -- combines distance-weighted selection with random perturbation

## Datasets

| Dataset | N | D | Known K | Notes |
|---------|---|---|---------|-------|
| Iris | 150 | 4 | 3 | Sanity check; ground truth available |
| Wine | 178 | 13 | 3 | Mixed-scale features; scaling matters |
| Mall Customers | 200 | 3 | ~5 | Purpose-built clustering demo |
| Credit Card Fraud | 284,807 | 29 | unknown | PCA-anonymized features; large N |
| MNIST (sklearn) | 1,797 | 64 | 10 | High D relative to N |
| SUSY (subset) | 500,000 | 50 | — | Main scale benchmark |

## Key Findings

**KeOps delivers on memory efficiency.** At 500k points and 50 dimensions, KeOps ran the
full K sweep under 200 MB peak VRAM. The equivalent PyTorch approach would require over
1 GB per iteration.

**Aggressive tensorization does not substitute for kernel fusion.** `kmeans_totally_tensor`
batches all restarts simultaneously but is slower and more memory-intensive than KeOps at
scale. Parallelizing restarts multiplies memory pressure -- the exact problem KeOps avoids.

**Initialization matters more as K and D grow.** On small datasets, all strategies produce
similar results. On SUSY with K up to 30, k-means++ consistently reached lower inertia and
converged faster than random initialization.

## Running the Notebook

1. Open the notebook on [Kaggle](https://kaggle.com) with GPU acceleration enabled
   (Notebook Settings > Accelerator > GPU)
2. Run the setup cells (KeOps install and CUDA path configuration)
3. Choose a dataset cell and run it, then run the scaling cell
4. Run the implementation cells in order
5. Run the sweep and comparison cells for results

The notebook is self-contained. No local installation is required.

**Key packages used:** pykeops, torch, scikit-learn, numpy, matplotlib, pandas

## Notebook Structure

```
Setup                  -- KeOps install, CUDA path, memory diagnostics
Data Loading           -- Six dataset options; run one, then the scaling cell
Initialization         -- Four centroid initialization strategies
kmeans_keops           -- Main KeOps implementation with inline explanation
kmeans_totally_tensor  -- Batched-restart experimental implementation
kmeans_sklearn         -- scikit-learn baseline
kmeans_tensor          -- Plain PyTorch baseline
Sweep & Benchmarking   -- Run any implementation across a K range
ARI Comparison         -- Correctness verification across implementations
Visualization          -- Elbow curve, inertia drop, timing, scatter plots
KeOps vs FAISS         -- Discussion of when each library is the right choice
Conclusions            -- Summary of findings
```

## Related

- [KeOps documentation](https://www.kernel-operations.io/keops/)
- [KeOps benchmarks](https://www.kernel-operations.io/keops/_auto_benchmarks/benchmark_KNN.html)
- [FAISS](https://github.com/facebookresearch/faiss)
