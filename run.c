// #include <conio.h>
#include <ctype.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

typedef struct {
    int dim;
    int n_layers;
    int vocab_size;
    int d_state;
    int d_conv;
    int expand;
    int dt_rank;
    int max_seq_len;
} Config;

typedef struct {
    float *token_embedding_table;
    float *rms_weight;
    float *in_proj_weight;
    float *conv1d_weight;
    float *conv1d_bias;
    float *x_proj_weight;
    float *dt_proj_weight;
    float *dt_proj_bias;
    float *out_proj_weight;
    float *A_log;
    float *D;
    float *rms_final_weight;
    float *wcls;
} MambaWeights;

typedef struct {
    // current wave of activations
    float *x;       // activation at current time stamp (dim,)
    float *xb;      // same, but inside a residual branch (dim,)
    float *xb2;     // an additional buffer just for convenience (dim,)
    float *hb;      // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q;       // query (dim,)
    float *k;       // key (dim,)
    float *v;       // value (dim,)
    float *att;     // buffer for scores/attention values (n_heads, seq_len)
    float *logits;  // output logits
    // kv cache
    float *key_cache;    // (layer, seq_len, dim)
    float *value_cache;  // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config;         // the hyperparameters of the architecture (the blueprint)
    MambaWeights weights;  // the weights of the model
    RunState state;        // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;             // file descriptor for memory mapping
    float *data;        // memory mapped data pointer
    ssize_t file_size;  // size of the checkpoint file in bytes
} Mamba;

void malloc_run_state(RunState *s, Config *p) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));

    // s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState *s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(MambaWeights *w, Config *p, float *ptr, bool shared_weights) {
    int d_inner = p->dim * p->expand;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_weight = ptr;
    ptr += n_layers * p->dim;
    w->in_proj_weight = ptr;
    ptr += n_layers * p->dim * (d_inner * 2);
    w->conv1d_weight = ptr;
    ptr += n_layers * (p->d_conv * d_inner);
    w->conv1d_bias = ptr;
    ptr += n_layers * (1 * d_inner);
    w->x_proj_weight = ptr;
    ptr += n_layers * d_inner * (p->dt_rank + p->d_state * 2);
    w->dt_proj_weight = ptr;
    ptr += n_layers * d_inner * p->dt_rank;
    w->dt_proj_bias = ptr;
    ptr += n_layers * d_inner * 1;
    w->out_proj_weight = ptr;
    ptr += n_layers * d_inner * p->dim;
    w->A_log = ptr;
    ptr += n_layers * d_inner * p->d_state;
    w->D = ptr;
    ptr += n_layers * d_inner;
    w->rms_final_weight = ptr;
    ptr += p->dim * 1;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char *checkpoint, Config *config, MambaWeights *weights, int *fd, float **data, ssize_t *file_size) {
    FILE *file = fopen(checkpoint, "rb");
    uint32_t file_identifier;
    int version;
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // Read in file identifier
    if (fread(&file_identifier, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr, "Error reading header %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // Read in file version
    if (fread(&version, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error reading header %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) {
        fprintf(stderr, "Error reading content %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    bool shared_weights;
    if (fread(&shared_weights, sizeof(bool), 1, file) != 1) {
        fprintf(stderr, "Error reading content %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);  // move file pointer to end of file
    *file_size = ftell(file);  // get the file size, in bytes
    fclose(file);

    *fd = open(checkpoint, O_RDONLY);  // open in read only mode
    if (*fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    float *weights_ptr = *data + 256 / sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Mamba *t, char *checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_mamba(Mamba *t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) {
        munmap(t->data, t->file_size);
    }
    if (t->fd != -1) {
        close(t->fd);
    }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Mamba

void rmsnorm(float *o, float *x, float *weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float *x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float *xout, float *x, float *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    // #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void conv1d(float *out, float *x, float *w, int n, int d, int k) {
    // x: input signal of length n
    // w: kernel of length k
    // d: number of filters
    // out: output signal of length n - k + 1

    for (int i = 0; i < d; i++) {              // Loop over filters
        for (int j = 0; j < n - k + 1; j++) {  // Loop over output positions
            float val = 0.0f;
            for (int m = 0; m < k; m++) {  // Loop over kernel elements
                val += x[j + m] * w[i * k + m];
            }
            out[i * (n - k + 1) + j] = val;
        }
    }
}

void conv1d_groups_padding(float *out, float *x, float *w, int n, int d, int k, int n_groups, int pad) {
    // x: input signal of length n
    // w: kernel of length k * d / n_groups
    // d: number of filters
    // n_groups: number of groups
    // pad: padding on each side
    // out: output signal of length n - k + 1 + 2 * pad

    int ch_per_group = d / n_groups;  // Channels per group
    int padded_n = n + 2 * pad;       // Padded input length

    // Pad the input signal with zeros
    float padded_x[padded_n];
    for (int i = 0; i < pad; i++) {
        padded_x[i] = 0.0f;
        padded_x[padded_n - i - 1] = 0.0f;
    }
    for (int i = 0; i < n; i++) {
        padded_x[i + pad] = x[i];
    }

    // Perform grouped convolution with padding
    for (int g = 0; g < n_groups; g++) {
        for (int i = g * ch_per_group; i < (g + 1) * ch_per_group; i++) {
            for (int j = 0; j < padded_n - k + 1; j++) {
                float val = 0.0f;
                for (int m = 0; m < k; m++) {
                    val += padded_x[j + m] * w[(i * k) + (g * ch_per_group * k) + m];
                }
                out[i * (padded_n - k + 1) + j] = val;
            }
        }
    }
}

void silu(float *x, int d) {
    // Swish-1: x * sigmoid(x)
    for (int i = 0; i < d; i++) {
        float val = x[i];
        val *= (1.0f / (1.0f + expf(-val)));
        x[i] = val;
    }
}

void negexp(float *x, int d) {
    for (int i = 0; i < d; i++) {
        x[i] = -expf(x[i]);
    }
}

void softplus(float *x, int d) {
    for (int i = 0; i < d; i++) {
        x[i] = logf(1.0f + expf(x[i]));
    }
}

// float *forward(Mamba *mamba, int token, int pos) {
//     // a few convenience variables
//     Config *p = &mamba->config;
//     MambaWeights *w = &mamba->weights;
//     RunState *s = &mamba->state;
//     float *x = s->x;
//     int dim = p->dim;
//     int d_state = p->d_state;
//     int d_conv = p->d_conv;
//     int expand = p->expand;
//     int d_inner = p->dim * p->expand;

//     // copy the token embedding into x
//     float *content_row = w->token_embedding_table + token * dim;
//     memcpy(x, content_row, dim * sizeof(*x));

//     // forward all the layers
//     for (unsigned long long l = 0; l < p->n_layers; l++) {
//         // attention rmsnorm
//         rmsnorm(s->xb, x, w->rms_weight + l * dim, dim);

//         matmul(s->q, s->xb, w->in_proj_weight + l * dim * (d_inner * 2), d_inner * 2, dim);

//         s->res = s->q + d_inner;

//         // conv1d_groups_padding(float *out, float *x, float *w, int n, int d, int k, int n_groups, int pad)
//         conv1d_groups_padding(s->xb, s->q, w->conv1d_weight + l * d_conv * d_inner, d_inner, d_conv, d_conv, d_inner, d_conv - 1);
//         for (int i = 0; i < d_inner; i++) {
//             s->xb[i] += w->conv1d_bias[l * d_inner + i];
//         }
//         silu(s->xb, d_inner);

//         s->A = negexp(w->A_log + l * d_inner * d_state, d_inner * d_state);
//         s->D = w->D + l * d_inner;

//         matmul(s->delta, s->xb, w->x_proj_weight + l * d_inner * (p->dt_rank + d_state * 2), p->dt_rank + d_state * 2, d_inner);
//         s->B = s->delta + p->dt_rank;
//         s->C = s->B + d_state;

//         softplus(s->delta, p->dt_rank);

//         // selective scan here
//         for (int k = 0; k < p->dt_rank + d_state * 2; k++) {
//             for (int i = 0; i < d_inner; i++) {
//                 s->deltaA[k][i] = s->delta[k] * s->A[k];
//             }
//         }

//         for (int i = 0; k < p->dt_rank + d_state * 2; i++) {
//             for (int j = 0; i < d_inner; j++) {
//                 s->deltaB_u[i][j] = s->delta[j] * s->B[j] * s->u[j];
//             }
//         }

//         for (int i = 0; i < d_inner; i++) {
//             s->xdelta[i][] =
//         }

//         // multihead attention. iterate over all heads
//         int h;
//         // #pragma omp parallel for private(h)
//         for (h = 0; h < p->n_heads; h++) {
//             // get the query vector for this head
//             float *q = s->q + h * head_size;
//             // attention scores for this head
//             float *att = s->att + h * p->seq_len;
//             // iterate over all timesteps, including the current one
//             for (int t = 0; t <= pos; t++) {
//                 // get the key vector for this head and at this timestep
//                 float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
//                 // calculate the attention score as the dot product of q and k
//                 float score = 0.0f;
//                 for (int i = 0; i < head_size; i++) {
//                     score += q[i] * k[i];
//                 }
//                 score /= sqrtf(head_size);
//                 // save the score to the attention buffer
//                 att[t] = score;
//             }

//             // softmax the scores to get attention weights, from 0..pos inclusively
//             softmax(att, pos + 1);

//             // weighted sum of the values, store back into xb
//             float *xb = s->xb + h * head_size;
//             memset(xb, 0, head_size * sizeof(float));
//             for (int t = 0; t <= pos; t++) {
//                 // get the value vector for this head and at this timestep
//                 float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
//                 // get the attention weight for this timestep
//                 float a = att[t];
//                 // accumulate the weighted value into xb
//                 for (int i = 0; i < head_size; i++) {
//                     xb[i] += a * v[i];
//                 }
//             }
//         }

//         // final matmul to get the output of the attention
//         matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

//         // residual connection back into x
//         for (int i = 0; i < dim; i++) {
//             x[i] += s->xb2[i];
//         }

//         // ffn rmsnorm
//         rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

//         // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
//         // first calculate self.w1(x) and self.w3(x)
//         matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
//         matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

//         // SwiGLU non-linearity
//         for (int i = 0; i < hidden_dim; i++) {
//             float val = s->hb[i];
//             // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
//             val *= (1.0f / (1.0f + expf(-val)));
//             // elementwise multiply with w3(x)
//             val *= s->hb2[i];
//             s->hb[i] = val;
//         }

//         // final matmul to get the output of the ffn
//         matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

//         // residual connection
//         for (int i = 0; i < dim; i++) {
//             x[i] += s->xb[i];
//         }
//     }

//     // final rmsnorm
//     rmsnorm(x, x, w->rms_final_weight, dim);

//     // classifier into logits
//     matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
//     return s->logits;
// }
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];  // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;  // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    // fprintf(stderr, "%s", tokenizer_path);
    if (fread(&t->max_token_length, sizeof(int32_t), 1, file) != 1) {
        fprintf(stderr, "failed read509\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read513, %ld\n", FLT_MAX);
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int32_t), 1, file) != 1) {
            fprintf(stderr, "failed read517\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "failed read522\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';  // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) {
        return;
    }
    if (piece[0] == '\0') {
        return;
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;  // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str};  // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=0) token, if desired
    if (bos) tokens[(*n_tokens)++] = 0;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""

    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    // if (text[0] != '\0') {
    //     int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    //     tokens[(*n_tokens)++] = dummy_prefix;
    // }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;  // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +2 is here because the first 2 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 2
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 2 - 33;
            }
        }
        str_len = 0;  // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;  // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;  // token length decreased
    }

    // add optional EOS (=0) token, if desired
    if (eos) tokens[(*n_tokens)++] = 0;

    free(str_buffer);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

void print_config(Config *config) {
    printf("\ndim = %d\n", config->dim);
    printf("n_layers = %d\n", config->n_layers);
    printf("vocab_size = %d\n", config->vocab_size);
    printf("d_state = %d\n", config->d_state);
    printf("d_conv = %d\n", config->d_conv);
    printf("expand = %d\n", config->expand);
    printf("dt_rank = %d\n", config->dt_rank);
    printf("max_seq_len = %d\n", config->max_seq_len);
}

void apply_chat_template(char *prompt, int user, int system) {
    char *temp = (char *)malloc(strlen(prompt) + 1);
    strcpy(temp, prompt);
    temp[strlen(prompt)] = '\0';
    free(prompt);
    if (user) {
        prompt = (char *)malloc(strlen(temp) + 1 + 9);
        strcpy(prompt, "<|user|>");
    } else if (system) {
        prompt = (char *)malloc(strlen(temp) + 1 + 11);
        strcpy(prompt, "<|system|>");
    } else {
        prompt = (char *)malloc(strlen(temp) + 1 + 14);
        strcpy(prompt, "<|assistant|>");
    }
    strcpy(prompt, temp);
    free(temp);
    temp = NULL;
    prompt[strlen(prompt)] = '\0';
}

int main(int argc, char **argv) {
    char *checkpoint_path = "mamba-130m.bin";
    Config config;
    MambaWeights weights;
    int fd;
    float *data;
    int64_t file_size;
    read_checkpoint(checkpoint_path, &config, &weights, &fd, &data, &file_size);
    print_config(&config);
    printf("Finished reading\n");
    Tokenizer tokenizer;
    // There are 50010 merges
    build_tokenizer(&tokenizer, "tokenizer.bin", 50009);

    char *text;
    text = (char *)malloc(33 * sizeof(char));
    strcpy(text, "<|user|>\nhello world! I am here.");
    text[32] = '\0';
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(text) + 3) * sizeof(int));  // +3 for '\0', ?BOS, ?EOS
    encode(&tokenizer, text, 1, 1, prompt_tokens, &num_prompt_tokens);
    printf("\nPrint 'hello world!\tI am here.' in tokens: \n");
    for (int i = 0; i < num_prompt_tokens; i++) {
        printf("%d %s\n", prompt_tokens[i], prompt_tokens[i] >= 0 ? tokenizer.vocab[prompt_tokens[i]] : "null");
    }
    printf("\n");
    return 0;
}