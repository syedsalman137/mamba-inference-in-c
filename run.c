#include <conio.h>
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
} MambaWeights;

typedef struct {
} Runstate;

typedef struct {
} Mamba;

void read_checkpoint(char *checkpoint, Config *config) {
    FILE *file = fopen(checkpoint, "rb");
    uint32_t magic_no;
    int version;
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // Read in header
    if (fread(&magic_no, sizeof(uint32_t), 1, file) != 1 || fread(&version, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr, "Error reading header %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) {
        fprintf(stderr, "Error reading content %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
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

int main(int argc, char **argv) {
    char *checkpoint_path = "mamba-130m.bin";
    Config config;
    read_checkpoint(checkpoint_path, &config);
    print_config(&config);
    printf("Finished reading\n");
    getch();
    return 0;
}