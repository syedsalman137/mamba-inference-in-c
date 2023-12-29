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
    float *x;    // activation at current time stamp (dim,)
    float *xb;   // same, but inside a residual branch (dim,)
    float *xb2;  // an additional buffer just for convenience (d_inner,)
    float *in_proj_out;
    float *out_proj_out;
    float *conv_out;
    float *logits;  // output logits
    // kv cache
    float *conv_state;
    float *x_ssm;
    float *A;
    float *x_proj_out;
    float *delta;
    float *deltaA;
    float *deltaB_u;
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
    int d_inner = p->dim * p->expand;
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(d_inner, sizeof(float));

    // s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->in_proj_out = calloc(d_inner * 2, sizeof(float));
    s->conv_state = calloc(p->n_layers * (p->max_seq_len + p->d_conv - 1) * d_inner, sizeof(float));
    s->conv_out = calloc(d_inner, sizeof(float));
    s->x_ssm = calloc(p->n_layers * d_inner * p->d_state, sizeof(float));
    s->out_proj_out = calloc(p->dim, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->A = calloc(d_inner * p->d_state, sizeof(float));
    s->x_proj_out = calloc(p->dt_rank + (p->d_state * 2), sizeof(float));
    s->delta = calloc(d_inner, sizeof(float));
    s->deltaA = calloc(d_inner * p->d_state, sizeof(float));
    s->deltaB_u = calloc(d_inner * p->d_state, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->in_proj_out || !s->conv_state || !s->conv_out || !s->x_ssm || !s->out_proj_out || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState *s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->in_proj_out);
    free(s->conv_state);
    free(s->conv_out);
    free(s->x_ssm);
    free(s->out_proj_out);
    free(s->logits);
    free(s->A);
    free(s->x_proj_out);
    free(s->delta);
    free(s->deltaA);
    free(s->deltaB_u);
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

    // read in the shared weights (between input embedding and linear output) flag
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

    // First 256 bytes contain headers, so skip them
    float *weights_ptr = *data + 256 / sizeof(float);

    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_mamba(Mamba *t, char *checkpoint_path) {
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

void silu(float *x, int d) {
    // Swish: x * sigmoid(x)
    for (int i = 0; i < d; i++) {
        float val = x[i];
        val *= (1.0f / (1.0f + expf(-val)));
        x[i] = val;
    }
}

void negexp(float *out, float *x, int d) {
    for (int i = 0; i < d; i++) {
        out[i] = -expf(x[i]);
    }
}

void softplus(float *x, int d) {
    for (int i = 0; i < d; i++) {
        x[i] = (float)log(1.0 + exp((double)x[i]));
    }
}

void ssm(float *y, unsigned long long l, RunState *s, Config *p, MambaWeights *w) {
    int dim = p->dim;
    int d_state = p->d_state;
    int d_conv = p->d_conv;
    int expand = p->expand;
    int d_inner = p->dim * p->expand;
    int dt_rank = p->dt_rank;
    int max_seq_len = p->max_seq_len;

    // SSM here
    matmul(s->x_proj_out, s->conv_out, w->x_proj_weight + l * (dt_rank + 2 * d_state) * d_inner, d_inner, dt_rank + 2 * d_state);
    float *B = s->x_proj_out + dt_rank;
    float *C = B + d_state;

    float *D = w->D + l * d_inner;

    matmul(s->delta, s->x_proj_out, w->dt_proj_weight + l * d_inner * dt_rank, dt_rank, d_inner);
    for (int i = 0; i < d_inner; i++) {
        s->delta[i] += w->dt_proj_bias[i];
    }
    softplus(s->delta, d_inner);

    for (int i = 0; i < d_inner; i++) {
        for (int j = 0; j < d_state; j++) {
            s->deltaA[i * d_state + j] = s->delta[i] * s->A[i * d_state + j];
        }
    }

    for (int i = 0; i < d_inner; i++) {
        for (int j = 0; j < d_state; j++) {
            s->deltaB_u[i * d_state + j] = s->delta[i] * B[j] * s->conv_out[i];
        }
    }

    for (int i = 0; i < d_inner; i++) {
        for (int j = 0; j < d_state; j++) {
            s->x_ssm[l * d_inner * d_state + i * d_state + j] *= s->deltaA[i * d_state + j];
            s->x_ssm[l * d_inner * d_state + i * d_state + j] += s->deltaB_u[i * d_state + j];
            y[i] += s->x_ssm[l * d_inner * d_state + i * d_state + j] * C[j];
        }
        y[i] += s->conv_out[i] * D[i];
    }
}

float *forward(Mamba *mamba, int token, int pos) {
    // a few convenience variables
    Config *p = &mamba->config;
    MambaWeights *w = &mamba->weights;
    RunState *s = &mamba->state;
    float *x = s->x;
    int dim = p->dim;
    int d_state = p->d_state;
    int d_conv = p->d_conv;
    int expand = p->expand;
    int d_inner = p->dim * p->expand;
    int dt_rank = p->dt_rank;
    int max_seq_len = p->max_seq_len;

    // copy the token embedding into x
    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_weight + l * dim, dim);
        matmul(s->in_proj_out, s->xb, w->in_proj_weight + l * dim * (d_inner * 2), dim, d_inner * 2);
        float *res = s->in_proj_out + d_inner;

        // set the conv state
        unsigned long long pos_idx = l * (max_seq_len + d_conv - 1) * d_inner + (pos + d_conv - 1) * d_inner;
        for (int i = 0; i < d_inner; i++) {
            float val = s->in_proj_out[i];
            s->conv_state[pos_idx + i] = val;
        }

        // Convolve
        unsigned long long layer_idx = l * (max_seq_len + d_conv - 1) * d_inner;
        for (int i = 0; i < d_inner; i++) {
            for (int j = pos, j_conv = 0; j < pos + d_conv; j++, j_conv++) {
                s->conv_out[i] = w->conv1d_weight[l * d_inner * d_conv + i * d_conv + j_conv] * s->conv_state[layer_idx + j * d_inner + i];
            }
        }

        // Conv bias
        for (int i = 0; i < d_inner; i++) {
            s->conv_out[i] += w->conv1d_bias[l * d_inner + i];
        }

        silu(s->conv_out, d_inner);

        float *y = calloc(d_inner, sizeof(float));
        ssm(y, l, s, p, w);

        for (int i = 0; i < d_inner; i++) {
            float val = res[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            y[i] *= val;
        }

        matmul(s->out_proj_out, y, w->out_proj_weight + l * dim * d_inner, d_inner, dim);
        for (int i = 0; i < dim; i++) {
            // Residual connection
            x[i] += s->out_proj_out[i];
        }

        fprintf(stderr, "%lld out of %d\n", l, p->n_layers);
        free(y);
    }
    // final rms norm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}
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

    // add_dummy_prefix is false by default
    // If you want to enable it, you can uncomment the following code
    // However, doing so will affect the quality of model outputs

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
            // TODO: Explain '-33'
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

void test_float(float f) {
    if (isnan(f)) {
        printf("nan\n");
    } else if (isinf(f)) {
        printf("inf\n");
    }
}

void test(Mamba *m) {
    MambaWeights *w = &m->weights;
    Config *p = &m->config;
    int d_inner = p->dim * p->expand;
    for (int i = 0; i < p->vocab_size * p->dim; i++) {
        test_float(w->token_embedding_table[i]);
    }
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        printf("%lld\n", l);
        for (int i = 0; i < p->dim; i++) {
            test_float(w->rms_weight[i + l * p->dim]);
        }

        for (int i = 0; i < p->dim * (d_inner * 2); i++) {
            test_float(w->in_proj_weight[i + l * p->dim * (d_inner * 2)]);
        }

        for (int i = 0; i < p->d_conv * d_inner; i++) {
            test_float(w->conv1d_weight[i + l * p->d_conv * d_inner]);
        }
        for (int i = 0; i < d_inner; i++) {
            test_float(w->conv1d_bias[i + l * d_inner]);
        }
        for (int i = 0; i < d_inner * (p->dt_rank + p->d_state * 2); i++) {
            test_float(w->x_proj_weight[i + l * d_inner * (p->dt_rank + p->d_state * 2)]);
        }
        for (int i = 0; i < d_inner * p->dt_rank; i++) {
            test_float(w->dt_proj_weight[i + l * d_inner * p->dt_rank]);
        }

        for (int i = 0; i < d_inner; i++) {
            test_float(w->dt_proj_bias[i + l * d_inner]);
        }
        for (int i = 0; i < d_inner * p->dim; i++) {
            test_float(w->out_proj_weight[i + l * d_inner * p->dim]);
        }
        for (int i = 0; i < d_inner * p->d_state; i++) {
            test_float(w->A_log[i + l * d_inner * p->d_state]);
        }
        for (int i = 0; i < d_inner; i++) {
            test_float(w->D[i + l * d_inner]);
        }
    }
    for (int i = 0; i < p->dim * 1; i++) {
        test_float(w->rms_final_weight[i]);
    }
}

int main(int argc, char **argv) {
    char *checkpoint_path = "mamba-130m.bin";
    Mamba mamba;
    build_mamba(&mamba, "mamba-130m.bin");
    // test(&mamba);
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

    float *logits = forward(&mamba, prompt_tokens[3], 0);
    for (int i = 0; i < mamba.config.vocab_size; i++) {
        printf("%f ", logits[i]);
    }
    free(text);
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_mamba(&mamba);
    return 0;
}