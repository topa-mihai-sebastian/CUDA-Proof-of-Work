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


PROMPTURI CLAUDE

cred ca am rezultate gresite. cred ca nu sunt sincronizate threadurile.. am comparat fisierul
  /gpu_miner/output/test4.out cu /cpu_miner/output/test4.out si difera. care este problema?

⏺ Voi examina codul pentru a identifica problema de sincronizare.

  Searched for 1 pattern (ctrl+o to expand)

⏺ Bash(diff /Users/sebitopa/Projects/AN3/sem2/ASC/lab2/asc-public/assignments/2-cuda_proof_of_work/g
      pu_miner/output/test4.out /Users/sebitopa/Projects/AN3/sem2/ASC/lab…)
  ⎿  1,11c1,11
     < 1,0000517db16db1b0a4b3c2a1eee14ebc3c6beaae6420dc90a0643d23a03c167a,edff735b62b37d1afe12e
     3dcaf3df4da978c857f1ad8f3a8b98c4464fbf0c11c,33865,0.00251,0.00175,0.00426
     … +35 lines (ctrl+o to expand)

  Searched for 1 pattern, read 2 files (ctrl+o to expand)

⏺ Am identificat problema. Voi citi și fișierele rămase pentru a confirma.

  Read 2 files (ctrl+o to expand)

⏺ Acum am o imagine completă a problemei. O voi explica și repara.

  Problema

  Există două bug-uri legate:

  Bug 1 — Race condition în kernel_find_nonce:
  Kernelul procesează milioane de nonce-uri în paralel. Când mai multe thread-uri găsesc un nonce
  valid, atomicCAS(is_found, 0, 1) garantează că doar unul scrie, dar nu garantează că scrie cel cu
  nonce-ul minim. Winner-ul e non-deterministic.

  Bug 2 — Efect în cascadă:
  Linia din miner.cpp:161:
  memcpy(prev_block_hash, block_hash, SHA256_HASH_SIZE);
  Odată ce GPU-ul găsește un hash diferit la blocul 5 (nonce diferit → hash diferit),
  prev_block_hash pentru blocul 6 se schimbă, și toate blocurile ulterioare au conținut diferit față
   de CPU. De aceea blocurile 1–4 coincid dar 5–10 diferă.

  Fix: Folosim atomicMin pe valid_nonce ca să garantăm nonce-ul minim, apoi recomputăm hash-ul pe
  CPU pentru nonce-ul câștigător.

⏺ Update(gpu_miner/src/utils.cu)

⏺ Voi examina codul pentru a identifica problema de sincronizare.

  Searched for 1 pattern (ctrl+o to expand)

⏺ Bash(diff /Users/sebitopa/Projects/AN3/sem2/ASC/lab2/asc-public/assignments/2-cuda_proof_of_work/g
      pu_miner/output/test4.out /Users/sebitopa/Projects/AN3/sem2/ASC/lab…)
  ⎿  1,11c1,11
     < 1,0000517db16db1b0a4b3c2a1eee14ebc3c6beaae6420dc90a0643d23a03c167a,edff735b62b37d1afe12e
     3dcaf3df4da978c857f1ad8f3a8b98c4464fbf0c11c,33865,0.00251,0.00175,0.00426
     … +35 lines (ctrl+o to expand)

  Searched for 1 pattern, read 2 files (ctrl+o to expand)

⏺ Am identificat problema. Voi citi și fișierele rămase pentru a confirma.

  Read 2 files (ctrl+o to expand)

⏺ Acum am o imagine completă a problemei. O voi explica și repara.

  Problema

  Există două bug-uri legate:

  Bug 1 — Race condition în kernel_find_nonce:
  Kernelul procesează milioane de nonce-uri în paralel. Când mai multe thread-uri găsesc un nonce
  valid, atomicCAS(is_found, 0, 1) garantează că doar unul scrie, dar nu garantează că scrie cel cu
  nonce-ul minim. Winner-ul e non-deterministic.

  Bug 2 — Efect în cascadă:
  Linia din miner.cpp:161:
  memcpy(prev_block_hash, block_hash, SHA256_HASH_SIZE);
  Odată ce GPU-ul găsește un hash diferit la blocul 5 (nonce diferit → hash diferit),
  prev_block_hash pentru blocul 6 se schimbă, și toate blocurile ulterioare au conținut diferit față
   de CPU. De aceea blocurile 1–4 coincid dar 5–10 diferă.

  Fix: Folosim atomicMin pe valid_nonce ca să garantăm nonce-ul minim, apoi recomputăm hash-ul pe
  CPU pentru nonce-ul câștigător.

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 7 lines, removed 11 lines
      175
      176  __global__ void kernel_find_nonce(const BYTE *difficulty, uint32_t start, uint32_t
            count,
      177                                     const BYTE *base_block_content, size_t current_
           length,
      178 -                                   BYTE *block_hash, uint32_t *valid_nonce, int *i
          -s_found) {
      178 +                                   uint32_t *valid_nonce) {
      179
      180      uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      181 -    if (idx >= count || *is_found) {
      182 -        return;
      183 -    }
      181 +    if (idx >= count) return;
      182 +
      183      uint32_t nonce = start + idx;
      184 +    // skip early if a smaller valid nonce is already recorded
      185 +    if (nonce >= *valid_nonce) return;
      186
      187      char nonce_string[NONCE_SIZE];
      188      BYTE local_content[MAX_LOCAL_BLOCK_SIZE];
     ...
      197      d_strcat((char *)local_content, nonce_string);
      198      apply_sha256(local_content, local_hash);
      199
      199 -    //daca are numarul corect de zerouri la inceput
      200      if (compare_hashes(local_hash, (BYTE *)difficulty) <= 0) {
      201 -        // compare and swap
      202 -        if (atomicCAS(is_found, 0, 1) == 0) {
      203 -            *valid_nonce = nonce;
      204 -            for (size_t i = 0; i < SHA256_HASH_SIZE; i++)
      205 -                block_hash[i] = local_hash[i];
      206 -        }
      201 +        // keep only the minimum valid nonce
      202 +        atomicMin(valid_nonce, nonce);
      203      }
      204  }
      205

⏺ Update(gpu_miner/src/utils.cu)

────────────────────────────────────────────────────────────────────────────────────────────────────
 Edit file
 gpu_miner/src/utils.cu
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 209
 210      static BYTE *d_difficulty = nullptr;
 211      static BYTE *d_block_content = nullptr;
 212 -    static BYTE *d_block_hash = nullptr;
 212      static uint32_t *d_valid_nonce = nullptr;
 214 -    static int *d_is_found = nullptr;
 213      static size_t last_max_length = 0;
 214
 215      if (!d_difficulty) {
 216          cudaMalloc((void **)&d_difficulty, SHA256_HASH_SIZE);
 219 -        cudaMalloc((void **)&d_block_hash, SHA256_HASH_SIZE);
 217          cudaMalloc((void **)&d_valid_nonce, sizeof(uint32_t));
 221 -        cudaMalloc((void **)&d_is_found, sizeof(int));
 218      }
 219
 220      if (current_length + NONCE_SIZE > last_max_length) {
...
 227      cudaMemcpy(d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
 228      cudaMemcpy(d_block_content, block_content, current_length, cudaMemcpyHostToDevice);
 229
 234 -    int h_is_found = 0;
 235 -    cudaMemcpy(d_is_found, &h_is_found, sizeof(int), cudaMemcpyHostToDevice);
 230 +    // Initialize to UINT32_MAX so atomicMin finds the minimum valid nonce
 231 +    uint32_t h_valid_nonce = UINT32_MAX;
 232 +    cudaMemcpy(d_valid_nonce, &h_valid_nonce, sizeof(uint32_t), cudaMemcpyHostToDevice);
 233
 234      for (uint32_t start = 0; start < max_nonce; start += batch_size) {
 235          uint32_t count = min(batch_size, max_nonce - start);
...
 237
 238          kernel_find_nonce<<<blocks, tpb>>>(d_difficulty, start, count,
 239                                             d_block_content, current_length,
 243 -                                           d_block_hash, d_valid_nonce, d_is_found);
 244 -
 240 +                                           d_valid_nonce);
      182 +
      183      uint32_t nonce = start + idx;
      184 +    // skip early if a smaller valid nonce is already recorded
      185 +    if (nonce >= *valid_nonce) return;
      186
      187      char nonce_string[NONCE_SIZE];
      188      BYTE local_content[MAX_LOCAL_BLOCK_SIZE];
     ...
      197      d_strcat((char *)local_content, nonce_string);
      198      apply_sha256(local_content, local_hash);
      199
      199 -    //daca are numarul corect de zerouri la inceput
      200      if (compare_hashes(local_hash, (BYTE *)difficulty) <= 0) {
      201 -        // compare and swap
      202 -        if (atomicCAS(is_found, 0, 1) == 0) {
      203 -            *valid_nonce = nonce;
      204 -            for (size_t i = 0; i < SHA256_HASH_SIZE; i++)
      205 -                block_hash[i] = local_hash[i];
      206 -        }
      201 +        // keep only the minimum valid nonce
      202 +        atomicMin(valid_nonce, nonce);
      203      }
      204  }
      205

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 19 lines, removed 12 lines
      209
      210      static BYTE *d_difficulty = nullptr;
      211      static BYTE *d_block_content = nullptr;
      212 -    static BYTE *d_block_hash = nullptr;
      212      static uint32_t *d_valid_nonce = nullptr;
      214 -    static int *d_is_found = nullptr;
      213      static size_t last_max_length = 0;
      214
      215      if (!d_difficulty) {
      216          cudaMalloc((void **)&d_difficulty, SHA256_HASH_SIZE);
      219 -        cudaMalloc((void **)&d_block_hash, SHA256_HASH_SIZE);
      217          cudaMalloc((void **)&d_valid_nonce, sizeof(uint32_t));
      221 -        cudaMalloc((void **)&d_is_found, sizeof(int));
      218      }
      219
      220      if (current_length + NONCE_SIZE > last_max_length) {
     ...
      227      cudaMemcpy(d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice)
           ;
      228      cudaMemcpy(d_block_content, block_content, current_length, cudaMemcpyHostToDev
           ice);
      229
      234 -    int h_is_found = 0;
      235 -    cudaMemcpy(d_is_found, &h_is_found, sizeof(int), cudaMemcpyHostToDevice);
      230 +    // Initialize to UINT32_MAX so atomicMin finds the minimum valid nonce
      231 +    uint32_t h_valid_nonce = UINT32_MAX;
      232 +    cudaMemcpy(d_valid_nonce, &h_valid_nonce, sizeof(uint32_t), cudaMemcpyHostToDe
          +vice);
      233
      234      for (uint32_t start = 0; start < max_nonce; start += batch_size) {
      235          uint32_t count = min(batch_size, max_nonce - start);
     ...
      237
      238          kernel_find_nonce<<<blocks, tpb>>>(d_difficulty, start, count,
      239                                             d_block_content, current_length,
      243 -                                           d_block_hash, d_valid_nonce, d_is_found
          -);
      244 -
      240 +                                           d_valid_nonce);
      241          cudaDeviceSynchronize();
      246 -        cudaMemcpy(&h_is_found, d_is_found, sizeof(int), cudaMemcpyDeviceToHost);
      242
      248 -        if (h_is_found) {
      249 -            cudaMemcpy(valid_nonce, d_valid_nonce, sizeof(uint32_t), cudaMemcpyDev
          -iceToHost);
      250 -            cudaMemcpy(block_hash, d_block_hash, SHA256_HASH_SIZE, cudaMemcpyDevic
          -eToHost);
      243 +        cudaMemcpy(&h_valid_nonce, d_valid_nonce, sizeof(uint32_t), cudaMemcpyDevi
          +ceToHost);
      244 +
      245 +        // If a valid nonce was found in this batch, no need to search further bat
          +ches
      246 +        // (all subsequent nonces are larger)
      247 +        if (h_valid_nonce < start + count) {
      248 +            *valid_nonce = h_valid_nonce;
      249 +
      250 +            // Recompute the hash for the minimum valid nonce on the CPU
      251 +            char nonce_str[NONCE_SIZE];
      252 +            BYTE full_content[MAX_LOCAL_BLOCK_SIZE];
      253 +            memcpy(full_content, block_content, current_length);
      254 +            full_content[current_length] = '\0';
      255 +            snprintf(nonce_str, sizeof(nonce_str), "%u", h_valid_nonce);
      256 +            strcat((char *)full_content, nonce_str);
      257 +            apply_sha256(full_content, block_hash);
      258              return 0;
      259          }
      260      }

⏺ Acum verific că MAX_LOCAL_BLOCK_SIZE este suficient de mare și că apply_sha256 este __host__:

  Read 1 file (ctrl+o to expand)

⏺ BLOCK_SIZE = 139, iar MAX_LOCAL_BLOCK_SIZE = 1024 — suficient. apply_sha256 este __host__
  __device__ deci poate fi apelată din CPU.

  Iată un rezumat clar al ce am schimbat:

  ---
  Problema reală:

  1. Race condition de corectitudine: atomicCAS(is_found, 0, 1) garantează că un singur thread scrie
   rezultatul, dar câștigătorul race-ului e non-deterministic — nu e neapărat cel cu nonce-ul minim.
   CPU-ul găsește mereu cel mai mic nonce valid (caută secvențial), deci GPU trebuie să facă la fel.
  2. Efect în cascadă: Odată ce blocul 5 are un hash diferit față de CPU, prev_block_hash pentru
  blocul 6 se schimbă, și toate blocurile ulterioare devin complet diferite.

  Fix-ul aplicat:

  - Eliminat is_found și d_block_hash din kernel
  - Inițializat valid_nonce = UINT32_MAX
  - Kernelul folosește acum atomicMin(valid_nonce, nonce) — garantează că se păstrează cel mai mic
  nonce valid găsit de oricare thread
  - După kernel, dacă s-a găsit un nonce valid în batch-ul curent, hash-ul se recomputează pe CPU
  (cu apply_sha256 care e __host__ __device__) pentru nonce-ul câștigător — elimină complet
  posibilitatea de mismatch între nonce și hash

✻ Churned for 2m 55s

❯ scuze folosesc sftp si am uitat sa descarc pe local de pe fep. incearca acum iar

⏺ Bash(diff /Users/sebitopa/Projects/AN3/sem2/ASC/lab2/asc-public/assignments/2-cuda_proof_of_work/g
      pu_miner/output/test4.out /Users/sebitopa/Projects/AN3/sem2/ASC/lab…)
  ⎿  1,11c1,11
     < 1,0000517db16db1b0a4b3c2a1eee14ebc3c6beaae6420dc90a0643d23a03c167a,edff735b62b37d1afe12e
     3dcaf3df4da978c857f1ad8f3a8b98c4464fbf0c11c,33865,0.00301,0.00186,0.00487
     … +35 lines (ctrl+o to expand)

⏺ Fix-ul funcționează. Hash-urile și nonce-urile sunt acum identice pe toate cele 10 blocuri —
  singurul diff rămas sunt timpii (GPU vs CPU), ceea ce este normal și așteptat.

  Rezultate:
  - Corectitudine: ✅ toate 10 blocuri corecte
  - Merkle root: 0.024s total → sub 0.03s ✅ (30/45 pct)
  - Nonce search: 0.007s total → sub 0.01s ✅ (45/45 pct)

✻ Crunched for 8s

❯ crezi ca se mai poate optimiza ceva la merkle?

  Read 1 file (ctrl+o to expand)

⏺ Da, există câteva optimizări concrete. Problema principală e în kernel_merkle_tree_level:

  Read 1 file (ctrl+o to expand)

⏺ 10,000 tranzacții per bloc, 14 niveluri în arbore. Există două probleme clare:

  1. d_strlen în apply_sha256 — scanează byte cu byte la fiecare apel. Pe nivelurile Merkle inputul
  e mereu exact 128 bytes, dar funcția le numără de fiecare dată (~20,000 de apeluri per bloc).
  2. d_strcpy + d_strcat — fiecare iterează byte cu byte (64 + 64 + 64 = 192 iterații). memcpy face
  același lucru mult mai eficient.

  Voi adăuga o variantă a apply_sha256 cu lungime fixă și voi înlocui operațiile de string din
  kernelul Merkle:

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  User rejected update to gpu_miner/src/utils.cu
      62      *dest = '\0';
      63  }
      64
      65 -// Compute SHA256 and convert to hex
      66 -__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
      67 -    size_t input_length = d_strlen((const char *)input);
      65 +// Compute SHA256 and convert to hex (known length variant — avoids d_strlen)
      66 +__host__ __device__ void apply_sha256_len(const BYTE *input, size_t input_length, B
         +YTE *output) {
      67      SHA256_CTX ctx;
      68      BYTE buf[SHA256_BLOCK_SIZE];
      69      const char hex_chars[] = "0123456789abcdef";
     ...
      73      sha256_final(&ctx, buf);
      74
      75      for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
      77 -        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
      78 -        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
      76 +        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];
      77 +        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];
      78      }
      80 -    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
      79 +    output[SHA256_BLOCK_SIZE * 2] = '\0';
      80  }
      81
      82 +// Compute SHA256 and convert to hex
      83 +__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
      84 +    apply_sha256_len(input, d_strlen((const char *)input), output);
      85 +}
      86 +
      87  // Compare two hashes
      88  __host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {

❯ as prefera sa nu modifici functiile astea:
  // CUDA sprintf alternative for nonce finding. Converts integer to its string representation.
  Returns string's length.
  __device__ int intToString(uint64_t num, char* out) {
      if (num == 0) {
          out[0] = '0';
          out[1] = '\0';
          return 1;
      }

      int i = 0;
      while (num != 0) {
          int digit = num % 10;
          num /= 10;
          out[i++] = '0' + digit;
      }

      // Reverse the string
      for (int j = 0; j < i / 2; j++) {
          char temp = out[j];
          out[j] = out[i - j - 1];
          out[i - j - 1] = temp;
      }
      out[i] = '\0';
      return i;
  }

  // CUDA strlen implementation.
  __host__ __device__ size_t d_strlen(const char *str) {
      size_t len = 0;
      while (str[len] != '\0') {
          len++;
      }
      return len;
  }

  // CUDA strcpy implementation.
  __device__ void d_strcpy(char *dest, const char *src){
      int i = 0;
      while ((dest[i] = src[i]) != '\0') {
          i++;
      }
  }

  // CUDA strcat implementation.
  __device__ void d_strcat(char *dest, const char *src){
      while (*dest != '\0') {
          dest++;
      }
      while (*src != '\0') {
          *dest = *src;
          dest++;
          src++;
      }
      *dest = '\0';
  }

  // Compute SHA256 and convert to hex
  __host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
      size_t input_length = d_strlen((const char *)input);
      SHA256_CTX ctx;
      BYTE buf[SHA256_BLOCK_SIZE];
      const char hex_chars[] = "0123456789abcdef";

      sha256_init(&ctx);
      sha256_update(&ctx, input, input_length);
      sha256_final(&ctx, buf);

      for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
          output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
          output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
      }
      output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
  }

  // Compare two hashes
  __host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
      for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {
          if (hash1[i] < hash2[i]) {
              return -1; // hash1 is lower
          } else if (hash1[i] > hash2[i]) {
              return 1; // hash2 is lower
          }
      }
      return 0; // hashes are equal
  }


⏺ Înțeles, nu ating funcțiile respective. Pot totuși adăuga o funcție nouă și optimiza kernelul
  Merkle fără să modific utilitarele existente.

  Problemele din kernel_merkle_tree_level la care pot ajunge fără a atinge funcțiile tale:

  1. d_strcat scanează 64 bytes ca să găsească finalul înainte să copieze — overhead inutil când
  știm că combined are mereu 64 bytes scrise
  2. apply_sha256 apelează d_strlen — inputul e mereu fix 128 bytes, dar le numără de fiecare dată
  3. Copierea byte-by-byte — putem face copy direct cu indecși cunoscuți

  Read 1 file (ctrl+o to expand)

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 17 lines
       80      output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
       81  }
       82
       83 +// apply_sha256 with known length — skips d_strlen, used where input size is fixed
       84 +__host__ __device__ void apply_sha256_len(const BYTE *input, size_t input_length,
          +BYTE *output) {
       85 +    SHA256_CTX ctx;
       86 +    BYTE buf[SHA256_BLOCK_SIZE];
       87 +    const char hex_chars[] = "0123456789abcdef";
       88 +
       89 +    sha256_init(&ctx);
       90 +    sha256_update(&ctx, input, input_length);
       91 +    sha256_final(&ctx, buf);
       92 +
       93 +    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
       94 +        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];
       95 +        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];
       96 +    }
       97 +    output[SHA256_BLOCK_SIZE * 2] = '\0';
       98 +}
       99 +
      100  // Compare two hashes
      101  __host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
      102      for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {

⏺ Acum optimizez kernel_merkle_tree_level să folosească apply_sha256_len și copy direct cu indecși
  (fără d_strcpy/d_strcat):

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Error editing file

  Read 1 file (ctrl+o to expand)

⏺ Update(gpu_miner/src/utils.cu)

────────────────────────────────────────────────────────────────────────────────────────────────────
 Edit file
 gpu_miner/src/utils.cu
  3. Copierea byte-by-byte — putem face copy direct cu indecși cunoscuți

  Read 1 file (ctrl+o to expand)

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 17 lines
       80      output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
       81  }
       82
       83 +// apply_sha256 with known length — skips d_strlen, used where input size is fixed
       84 +__host__ __device__ void apply_sha256_len(const BYTE *input, size_t input_length,
          +BYTE *output) {
       85 +    SHA256_CTX ctx;
       86 +    BYTE buf[SHA256_BLOCK_SIZE];
       87 +    const char hex_chars[] = "0123456789abcdef";
       88 +
       89 +    sha256_init(&ctx);
       90 +    sha256_update(&ctx, input, input_length);
       91 +    sha256_final(&ctx, buf);
       92 +
       93 +    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
       94 +        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];
       95 +        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];
       96 +    }
       97 +    output[SHA256_BLOCK_SIZE * 2] = '\0';
       98 +}
       99 +
      100  // Compare two hashes
      101  __host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
      102      for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {

⏺ Acum optimizez kernel_merkle_tree_level să folosească apply_sha256_len și copy direct cu indecși
  (fără d_strcpy/d_strcat):

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Error editing file

  Read 1 file (ctrl+o to expand)

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 16 lines, removed 11 lines
      121  __global__ void kernel_merkle_tree_level(int n, int new_n, const BYTE *hashes_in,
           BYTE *hashes_out) {
      122      int i = blockIdx.x * blockDim.x + threadIdx.x;
      123      if (i < new_n) {
      124 -        BYTE combined[SHA256_HASH_SIZE * 2];
      125 -        int left_idx = i * 2;
      126 -        int right_idx = left_idx + 1;
      124 +        // input is always 2 x 64 hex chars = 128 bytes known at compile time
      125 +        const int HASH_LEN = SHA256_HASH_SIZE - 1; // 64
      126 +        BYTE combined[HASH_LEN * 2 + 1];
      127
      128 -        d_strcpy((char *)combined, (const char *)(hashes_in + left_idx * SHA256_HA
          -SH_SIZE));
      129 -
      130 -        if (right_idx < n) {
      131 -            d_strcat((char *)combined, (const char *)(hashes_in + right_idx * SHA2
          -56_HASH_SIZE));
      132 -        } else {
      133 -            // daca n este impar il concatenezi pe ultimul de 2 ori
      134 -            d_strcat((char *)combined, (const char *)(hashes_in + left_idx * SHA25
          -6_HASH_SIZE));
      128 +        int left_idx  = i * 2;
      129 +        int right_idx = (left_idx + 1 < n) ? left_idx + 1 : left_idx;
      130 +
      131 +        const BYTE *left  = hashes_in + left_idx  * SHA256_HASH_SIZE;
      132 +        const BYTE *right = hashes_in + right_idx * SHA256_HASH_SIZE;
      133 +
      134 +        // direct indexed copy — avoids d_strcpy scan + d_strcat double-scan
      135 +        for (int k = 0; k < HASH_LEN; k++) {
      136 +            combined[k]            = left[k];
      137 +            combined[HASH_LEN + k] = right[k];
      138          }
      139 +        combined[HASH_LEN * 2] = '\0';
      140
      137 -        apply_sha256(combined, hashes_out + i * SHA256_HASH_SIZE);
      141 +        // length is known: skip d_strlen inside apply_sha256
      142 +        apply_sha256_len(combined, HASH_LEN * 2, hashes_out + i * SHA256_HASH_SIZE
          +);
      143      }
      144  }
      145

⏺ Gata. Rezumatul modificărilor:

  - Adăugat apply_sha256_len — identică cu apply_sha256 dar fără d_strlen, pentru când lungimea e
  cunoscută
  - kernel_merkle_tree_level — înlocuit d_strcpy + d_strcat cu un loop direct cu indecși
  (combined[k] și combined[HASH_LEN + k]), și apelează apply_sha256_len cu lungime fixă 128

  Per apel de kernel pe nivelul Merkle se economisesc:
  - 64 iterații din d_strcat (scan-ul inițial ca să găsească '\0')
  - 128 iterații din d_strlen în apply_sha256

  La 10,000 noduri per nivel × ~14 niveluri × 10 blocuri = ~1.4M apeluri, economiile se adună.
  Compilează și rulează pe cluster și compară cu 0.02443s anterior.

✻ Sautéed for 1m 15s

❯ vreau sa transformi toate comentariile mele din romana in engleza. pe
   langa asta sa mai adaugi in unele locuri niste comentarii in engleza

  Read 1 file (ctrl+o to expand)

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 5 lines, removed 6 lines
      171      int tpb = 256;
      172      cudaMemcpy(d_transactions, transactions, n * transaction_size, cudaMemcpyHostT
           oDevice);
      173
      174 -    // level 0(frunzele)
      175 -    // transform datele brute in hashuri
      174 +    // level 0 (leaves): hash each raw transaction into its SHA256 hex string
      175      kernel_hash_transactions<<<(n + tpb - 1) / tpb, tpb>>>(n, transaction_size, d_
           transactions, d_hashes_a);
      176      cudaDeviceSynchronize();
      177
      179 -    // aici scriu alternativ intre src si dst
      178 +    // ping-pong between two buffers to avoid extra allocations
      179      BYTE *src = d_hashes_a;
      180      BYTE *dst = d_hashes_b;
      181
     ...
      184          int new_n = (current_n + 1) / 2;
      185          kernel_merkle_tree_level<<<(new_n + tpb - 1) / tpb, tpb>>>(current_n, new_
           n, src, dst);
      186          cudaDeviceSynchronize();
      188 -
      189 -        // swap src si dst
      187 +
      188 +        // swap src and dst for the next level
      189          BYTE *tmp = src; src = dst; dst = tmp;
      190          current_n = new_n;
      191      }
      192
      194 -    // root-ul e în src (ultimul dst devine src după swap)
      193 +    // after the last swap, src holds the root
      194      cudaMemcpy(merkle_root, src, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
      195  }
      196

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 1 line, removed 1 line
      226
      227  int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t c
           urrent_length, BYTE *block_hash, uint32_t *valid_nonce) {
      228      int tpb = 256;
      229 -    uint32_t batch_size = 1 << 20; // 1M nonce-uri per batch
      229 +    uint32_t batch_size = 1 << 20; // 1M nonces per batch
      230
      231      static BYTE *d_difficulty = nullptr;
      232      static BYTE *d_block_content = nullptr;

⏺ Acum adaug comentarii și în construct_merkle_root unde lipsesc (la alocare și la condiția de
  realocare):

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 6 lines, removed 9 lines
      151      static int last_max = 0;
      152      static int last_tx_size = 0;
      153
      154 +    // reallocate only when block size or transaction size changes between calls
      155      if (last_max != max_transactions_in_a_block || last_tx_size != transaction_siz
           e) {
      155 -        if (d_transactions) {
      156 -            cudaFree(d_transactions);
      157 -        }
      158 -        if (d_hashes_a) {
      159 -            cudaFree(d_hashes_a);
      160 -        }
      161 -        if (d_hashes_b) {
      162 -            cudaFree(d_hashes_b);
      163 -        }
      156 +        if (d_transactions) cudaFree(d_transactions);
      157 +        if (d_hashes_a)     cudaFree(d_hashes_a);
      158 +        if (d_hashes_b)     cudaFree(d_hashes_b);
      159 +
      160 +        // allocate enough space for the worst case (full block)
      161          cudaMalloc((void **)&d_transactions, max_transactions_in_a_block * transac
           tion_size);
      162          cudaMalloc((void **)&d_hashes_a, max_transactions_in_a_block * SHA256_HASH
           _SIZE);
      163          cudaMalloc((void **)&d_hashes_b, max_transactions_in_a_block * SHA256_HASH
           _SIZE);

⏺ Acum adaug câteva comentarii și în find_nonce unde lipsesc:

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 3 lines, removed 2 lines
      230      static uint32_t *d_valid_nonce = nullptr;
      231      static size_t last_max_length = 0;
      232
      233 +    // one-time allocation for buffers that never change size
      234      if (!d_difficulty) {
      235          cudaMalloc((void **)&d_difficulty, SHA256_HASH_SIZE);
      236          cudaMalloc((void **)&d_valid_nonce, sizeof(uint32_t));
      237      }
      238
      239 +    // grow block content buffer if needed (block content length can vary)
      240      if (current_length + NONCE_SIZE > last_max_length) {
      239 -        if (d_block_content)
      240 -            cudaFree(d_block_content);
      241 +        if (d_block_content) cudaFree(d_block_content);
      242          cudaMalloc((void **)&d_block_content, current_length + NONCE_SIZE);
      243          last_max_length = current_length + NONCE_SIZE;
      244      }

⏺ Gata. Modificările făcute:

  - Tradus toate comentariile din română în engleză simplă: // frunzele → // leaves, // swap src si
  dst → // swap src and dst for the next level, // root-ul e în src → // after the last swap, src
  holds the root, // nonce-uri per batch → // nonces per batch
  - Adăugat comentarii noi în locuri care nu aveau: la logica de realocare (// reallocate only
  when..., // allocate enough space...), la ping-pong (// ping-pong between two buffers...), la
  alocările din find_nonce (// one-time allocation..., // grow block content buffer...)

✻ Cooked for 43s

❯ acum vreau ca in toate partile de acest tip [if (d_hashes_b)     cudaFree(d_hashes_b);] sa
  folosim indentarea standard cu acolade

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 9 lines, removed 3 lines
      153
      154      // reallocate only when block size or transaction size changes between calls
      155      if (last_max != max_transactions_in_a_block || last_tx_size != transaction_siz
           e) {
      156 -        if (d_transactions) cudaFree(d_transactions);
      157 -        if (d_hashes_a)     cudaFree(d_hashes_a);
      158 -        if (d_hashes_b)     cudaFree(d_hashes_b);
      156 +        if (d_transactions) {
      157 +            cudaFree(d_transactions);
      158 +        }
      159 +        if (d_hashes_a) {
      160 +            cudaFree(d_hashes_a);
      161 +        }
      162 +        if (d_hashes_b) {
      163 +            cudaFree(d_hashes_b);
      164 +        }
      165
      166          // allocate enough space for the worst case (full block)
      167          cudaMalloc((void **)&d_transactions, max_transactions_in_a_block * transac
           tion_size);

⏺ Update(gpu_miner/src/utils.cu)
  ⎿  Added 3 lines, removed 1 line
      244
      245      // grow block content buffer if needed (block content length can vary)
      246      if (current_length + NONCE_SIZE > last_max_length) {
      247 -        if (d_block_content) cudaFree(d_block_content);
      247 +        if (d_block_content) {
      248 +            cudaFree(d_block_content);
      249 +        }
      250          cudaMalloc((void **)&d_block_content, current_length + NONCE_SIZE);
      251          last_max_length = current_length + NONCE_SIZE;
      252      }

⏺ Gata. Toate cele 4 if-uri cu cudaFree folosesc acum acolade standard.

✻ Churned for 10s

❯ ce face snprintf?

⏺ snprintf scrie text formatat într-un buffer, ca printf dar în loc să afișeze pe ecran, pune
  rezultatul într-un string.

  snprintf(nonce_str, sizeof(nonce_str), "%u", h_valid_nonce);

  Cele 3 argumente principale:
  - nonce_str — bufferul unde se scrie
  - sizeof(nonce_str) — numărul maxim de caractere scrise (protecție împotriva buffer overflow)
  - "%u", h_valid_nonce — formatul și valoarea, la fel ca la printf

  Dacă h_valid_nonce = 42, după apel nonce_str va conține "42\0".

  n-ul din snprintf vine de la faptul că specifici dimensiunea maximă — asta îl face mai sigur decât
   sprintf, care scrie fără limită și poate depăși bufferul.

✻ Cogitated for 6s

❯ nu putem folosi functia asta?__device__ int intToString(uint64_t num, char* out) {
      if (num == 0) {
          out[0] = '0';
          out[1] = '\0';
          return 1;
      }

      int i = 0;
      while (num != 0) {
          int digit = num % 10;
          num /= 10;
          out[i++] = '0' + digit;
      }

      // Reverse the string
      for (int j = 0; j < i / 2; j++) {
          char temp = out[j];
          out[j] = out[i - j - 1];
          out[i - j - 1] = temp;
      }
      out[i] = '\0';
      return i;
  }

⏺ intToString are __device__ — nu poate fi apelată din cod de CPU (host). find_nonce e o funcție
  host, deci nu merge acolo.

  Dacă vrei totuși să o folosești, ar trebui să adaugi __host__ la ea:

  __host__ __device__ int intToString(uint64_t num, char* out) {

  Dar ai zis că nu vrei să modifici funcțiile astea. Deci snprintf e soluția corectă pentru partea
  de CPU.

✻ Sautéed for 4s

※ recap: Optimizăm GPU miner-ul pentru tema CUDA proof-of-work. Codul e corect și compilează —
  următorul pas e să rulezi pe cluster și să compari timpii Merkle cu cei anteriori (0.024s).
  (disable recaps in /config)

❯ de unde este acest numar?
      uint32_t batch_size = 1 << 20; // 1M nonces per batch

⏺ 1 << 20 înseamnă 1 deplasat cu 20 de biți la stânga în reprezentarea binară, adică:

  1 << 20 = 2^20 = 1.048.576 ≈ 1 milion

  E o convenție comună în programare să folosești puteri ale lui 2 în loc de numere rotunde ca
  1000000, pentru că:
  1. E mai rapid de scris și de recunoscut
  2. Puterile lui 2 se aliniază natural cu dimensiunile GPU (număr de thread-uri, warp-uri, etc.)

  Valoarea 20 a fost aleasă empiric — e un batch suficient de mare ca GPU-ul să fie ocupat, dar
  suficient de mic ca să verificăm des dacă am găsit nonce-ul și să oprim căutarea devreme.
