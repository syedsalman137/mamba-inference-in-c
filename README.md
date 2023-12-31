# mamba-inference-in-c

This repo runs Mamba models in C. We use tokenizer of Zephyr family of models, available on Huggingface.

Runs about 20-25% slower 😔 than llama2.c when comparing mamba-130m vs llama-110m.

## Build and Run

Clone the repo on your device

```
git clone https://github.com/SalmanHabeeb/mamba-inference-in-c.git
```

and navigate to the folder:
```
cd mamba-inference-in-c
```

Build using
```
make runomp
```
Other build commands are listed inside [Makefile](https://github.com/SalmanHabeeb/mamba-inference-in-c/blob/main/Makefile)

Models can be downloaded from this [link](https://drive.google.com/drive/folders/1Ls-_WnTxlR9fvf8Mfii1B3Idf5M1jgES?usp=sharing) manually, or using gdown:

```
# For model
gdown https://drive.google.com/file/d/1cI6_LmfSuKLtgGNyOUbcQ2K_2h6a-KaL/view?usp=sharing --fuzzy

# For tokenizer
gdown https://drive.google.com/file/d/1qUjULatBdbrJaJqsJuTrtuWYt4G6GI2D/view?usp=sharing --fuzzy
```

Now run the model to generate text using:
```
OMP_NUM_THREADS=2 ./run "path/to/model"
```

and for chatting, use:
```
OMP_NUM_THREADS=2 ./run "path/to/model" -m chat
```

## Models

Also, models can be exported to required bin format by running

```
python export_model.py path/to/save/model --checkpoint "huggingface-chkpoint"
```

Also, tokenizer can be obtained by using:

```
python export_tokenizer.py path/to/save/tokenizer --tokenizer "huggingface-tokenizer"
```

## Attribution

- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) -- Basically this repository is unofficial fork of karpathy/llama2.c
- [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal) -- Inspired mamba implementation
- [havenhq/mamba-chat](https://github.com/havenhq/mamba-chat) -- For chat-finetuning of models
- [state-spaces/mamba](https://github.com/state-spaces/mamba)