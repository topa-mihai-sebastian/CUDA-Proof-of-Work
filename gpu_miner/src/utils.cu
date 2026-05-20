#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_LOCAL_BLOCK_SIZE 1024

// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
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

// apply_sha256 with known length
__host__ __device__ void apply_sha256_len(const BYTE *input, size_t input_length, BYTE *output) {
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0';
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

// take the real data and hash it
__global__ void kernel_hash_transactions(int n, int transaction_size, const BYTE *transactions, BYTE* hashes){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        apply_sha256(transactions + i * transaction_size, hashes + i * SHA256_HASH_SIZE);
    }
}

// hash(2 hashes combined)
__global__ void kernel_merkle_tree_level(int n, int new_n, const BYTE *hashes_in, BYTE *hashes_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < new_n) {
        // input is always 2 x 64 hex chars = 128 bytes known at compile time
        const int HASH_LEN = SHA256_HASH_SIZE - 1; // 64
        BYTE combined[HASH_LEN * 2 + 1];

        int left_idx  = i * 2;
        int right_idx = (left_idx + 1 < n) ? left_idx + 1 : left_idx;

        const BYTE *left  = hashes_in + left_idx  * SHA256_HASH_SIZE;
        const BYTE *right = hashes_in + right_idx * SHA256_HASH_SIZE;

        // direct indexed copy — avoids d_strcpy scan + d_strcat double-scan
        for (int k = 0; k < HASH_LEN; k++) {
            combined[k]            = left[k];
            combined[HASH_LEN + k] = right[k];
        }
        combined[HASH_LEN * 2] = '\0';

        // length is known: skip d_strlen inside apply_sha256
        apply_sha256_len(combined, HASH_LEN * 2, hashes_out + i * SHA256_HASH_SIZE);
    }
}

// TODO 1: Implement this function in CUDA
void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]) {
    static BYTE *d_transactions = nullptr;
    static BYTE *d_hashes_a = nullptr;
    static BYTE *d_hashes_b = nullptr;
    static int last_max = 0;
    static int last_tx_size = 0;

    // reallocate only when block size or transaction size changes between calls
    if (last_max != max_transactions_in_a_block || last_tx_size != transaction_size) {
        if (d_transactions) {
            cudaFree(d_transactions);
        }
        if (d_hashes_a) {
            cudaFree(d_hashes_a);
        }
        if (d_hashes_b) {
            cudaFree(d_hashes_b);
        }

        // allocate enough space for the worst case (full block)
        cudaMalloc((void **)&d_transactions, max_transactions_in_a_block * transaction_size);
        cudaMalloc((void **)&d_hashes_a, max_transactions_in_a_block * SHA256_HASH_SIZE);
        cudaMalloc((void **)&d_hashes_b, max_transactions_in_a_block * SHA256_HASH_SIZE);
        last_max = max_transactions_in_a_block;
        last_tx_size = transaction_size;
    }

    int tpb = 256;
    cudaMemcpy(d_transactions, transactions, n * transaction_size, cudaMemcpyHostToDevice);

    // level 0 (leaves): hash each raw transaction (raw data) into its SHA256 hex string
    kernel_hash_transactions<<<(n + tpb - 1) / tpb, tpb>>>(n, transaction_size, d_transactions, d_hashes_a);
    cudaDeviceSynchronize();

    // ping-pong between two buffers to avoid extra allocations
    BYTE *src = d_hashes_a;
    BYTE *dst = d_hashes_b;

    int current_n = n;
    while (current_n > 1) {
        int new_n = (current_n + 1) / 2;
        kernel_merkle_tree_level<<<(new_n + tpb - 1) / tpb, tpb>>>(current_n, new_n, src, dst);
        cudaDeviceSynchronize();

        // swap src and dst for the next level
        BYTE *tmp = src; src = dst; dst = tmp;
        current_n = new_n;
    }

    // after the last swap, src holds the root
    cudaMemcpy(merkle_root, src, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
}

__global__ void kernel_find_nonce(const BYTE *difficulty, uint32_t start, uint32_t count,
                                   const BYTE *base_block_content, size_t current_length,
                                   uint32_t *valid_nonce) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint32_t nonce = start + idx;
    // skip early if a smaller valid nonce is already recorded
    if (nonce >= *valid_nonce) return;

    char nonce_string[NONCE_SIZE];
    BYTE local_content[MAX_LOCAL_BLOCK_SIZE];
    BYTE local_hash[SHA256_HASH_SIZE];

    for (size_t i = 0; i < current_length; i++) {
        local_content[i] = base_block_content[i];
    }
    local_content[current_length] = '\0';

    intToString(nonce, nonce_string);
    d_strcat((char *)local_content, nonce_string);
    apply_sha256(local_content, local_hash);

    if (compare_hashes(local_hash, (BYTE *)difficulty) <= 0) {
        // keep only the minimum valid nonce
        atomicMin(valid_nonce, nonce);
    }
}

int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    int tpb = 256;
    uint32_t batch_size = 1 << 20; // 1M nonces per batch

    static BYTE *d_difficulty = nullptr;
    static BYTE *d_block_content = nullptr;
    static uint32_t *d_valid_nonce = nullptr;
    static size_t last_max_length = 0;

    // one-time allocation for buffers that never change size
    if (!d_difficulty) {
        cudaMalloc((void **)&d_difficulty, SHA256_HASH_SIZE);
        cudaMalloc((void **)&d_valid_nonce, sizeof(uint32_t));
    }

    // grow block content buffer if needed (block content length can vary)
    if (current_length + NONCE_SIZE > last_max_length) {
        if (d_block_content) {
            cudaFree(d_block_content);
        }
        cudaMalloc((void **)&d_block_content, current_length + NONCE_SIZE);
        last_max_length = current_length + NONCE_SIZE;
    }

    cudaMemcpy(d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_content, block_content, current_length, cudaMemcpyHostToDevice);

    // initialize to UINT32_MAX so atomicMin finds the minimum valid nonce
    uint32_t h_valid_nonce = UINT32_MAX;
    cudaMemcpy(d_valid_nonce, &h_valid_nonce, sizeof(uint32_t), cudaMemcpyHostToDevice);

    for (uint32_t start = 0; start < max_nonce; start += batch_size) {
        uint32_t count = min(batch_size, max_nonce - start);
        int blocks = (count + tpb - 1) / tpb;

        kernel_find_nonce<<<blocks, tpb>>>(d_difficulty, start, count,
                                           d_block_content, current_length,
                                           d_valid_nonce);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_valid_nonce, d_valid_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // if a valid nonce was found in this batch, no need to search further batches
        // (all subsequent nonces are larger)
        if (h_valid_nonce < start + count) {
            *valid_nonce = h_valid_nonce; // nonce minim

            // recompute the hash for the minimum valid nonce on the CPU
            char nonce_str[NONCE_SIZE];
            BYTE full_content[MAX_LOCAL_BLOCK_SIZE];
            memcpy(full_content, block_content, current_length);
            full_content[current_length] = '\0';
            snprintf(nonce_str, sizeof(nonce_str), "%u", h_valid_nonce); //converts to string
            strcat((char *)full_content, nonce_str); // append to the content
            apply_sha256(full_content, block_hash);
            return 0;
        }
    }

    return 1;
}


__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu() {
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);
}
