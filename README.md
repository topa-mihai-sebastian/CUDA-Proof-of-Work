Name: Topa Mihai-Sebastian

Group: 333CC

# Tema 2 - CUDA Proof of Work

## Organization

### General approach

The assignment asks us to parallelize two things on the GPU: building a Merkle tree from transactions, and finding a valid nonce for each block.

**Merkle tree (`construct_merkle_root`)**

The idea is simple: hash all transactions in parallel (level 0), then keep combining pairs of hashes level by level until only one hash is left (the root). Each level is one CUDA kernel launch. Two GPU buffers are used in a ping-pong fashion — one is the input, the other is the output, then they swap for the next level. This avoids any extra allocations per level.

```c
// level 0: hash each transaction
kernel_hash_transactions<<<...>>>(n, transaction_size, d_transactions, d_hashes_a);

// each level halves the number of nodes
while (current_n > 1) {
    kernel_merkle_tree_level<<<...>>>(current_n, new_n, src, dst);
    swap(src, dst);
    current_n = new_n;
}
```

An optimization was added for the Merkle level kernel: since every input is always exactly 128 bytes (two 64-char hex strings), we skip the `d_strlen` call by using `apply_sha256_len` with a known length. The string copy was also replaced with a direct indexed loop instead of `d_strcpy` + `d_strcat`, which avoids the overhead of scanning for the null terminator twice.

**Nonce search (`find_nonce`)**

The GPU tries many nonces at the same time instead of one by one. Nonces are split into batches of 1M (`1 << 20`). Inside each batch, every thread tests one nonce: it builds the block content with that nonce appended, computes SHA256, and checks if the hash is below the difficulty target.

The tricky part is making sure we find the **minimum** valid nonce, same as the CPU (which searches from 0 upward). The original approach used `atomicCAS` to let the first thread that finds a valid nonce "win", but that thread is not necessarily the one with the smallest nonce — it's just the fastest. This caused results to differ from the CPU reference.

The fix: initialize `valid_nonce = UINT32_MAX` and use `atomicMin` instead. Every thread that finds a valid nonce tries to update `valid_nonce` with its own nonce. Since `atomicMin` keeps the smallest value, by the end of the kernel we are guaranteed to have the minimum valid nonce — regardless of which threads finished first.

```c
if (compare_hashes(local_hash, difficulty) <= 0) {
    atomicMin(valid_nonce, nonce);  // keep the smallest valid nonce found
}
```

After the kernel, the hash for the winning nonce is recomputed once on the CPU. This is simpler and avoids storing the hash inside the kernel (which would require extra synchronization).

Batches stop early as soon as a valid nonce is found in the current batch, since all remaining batches would only contain larger nonces.

**Why batches at all?**

If we launched all nonces at once (up to `UINT32_MAX` = ~4 billion), we would need to allocate memory for all of them. Batches let us check for early exit after each 1M nonces, so if the answer is nonce 7978 we don't waste time computing the other 4 billion.

**Is the implementation efficient?**

It is reasonably efficient. The Merkle tree is fully parallelized and runs in ~0.02083s total. The nonce search runs in ~0.007s total. Both are well within the performance thresholds from the assignment. The GPU is roughly 70x faster than the CPU for this workload.

The main remaining overhead in the Merkle tree is the `cudaDeviceSynchronize()` call between each level — this is unavoidable since each level depends on the previous one. With 14 levels per block and 10 blocks, that is 140 synchronization points.

**Is the task useful?**

Yes. It shows a concrete example where GPU parallelism gives a large real-world speedup (proof-of-work mining is exactly the kind of embarrassingly parallel problem GPUs are built for). The nonce search is a good example of how to handle non-determinism in parallel programs correctly using atomics.

---

## Implementation

The full assignment is implemented:
- `construct_merkle_root` — fully parallelized with CUDA kernels
- `find_nonce` — fully parallelized with batched CUDA kernels

Extra additions beyond the spec:
- `apply_sha256_len` — a variant of `apply_sha256` that takes a known length and skips `d_strlen`. Used in `kernel_merkle_tree_level` where the input size is always fixed at 128 bytes.

No missing functionality. All 10 blocks in `test4` produce correct results matching the CPU reference.

**Difficulties:**
- The original `atomicCAS`-based nonce search produced correct-looking output (hashes with the right number of leading zeros) but returned a non-minimum nonce. This caused a cascade failure: once block 5 had a different hash than the CPU, every subsequent block had a different `prev_hash` and therefore different content, making all results after block 4 wrong.
- CUDA device functions (`__device__`) cannot be called from host code, so `intToString` could not be reused in the CPU-side hash recomputation after finding the minimum nonce.
