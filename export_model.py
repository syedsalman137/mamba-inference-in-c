import struct
import torch
from torch import nn
import argparse
import struct
import numpy as np
import gc
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


MAX_SEQUENCE_LENGTH = 256


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def serialize_int8(file, tensor):
    """writes one int8 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f"{len(d)}b", *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


def version2_export(model, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 2

    # let's first do some validation for this export type
    while model.config.d_model % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    weights = [
        model.backbone.embedding.weight,
        *[layer.mixer.in_proj.weight for layer in model.backbone.layers],
        *[layer.mixer.conv1d.weight for layer in model.backbone.layers],
        *[layer.mixer.conv1d.bias for layer in model.backbone.layers],
        *[layer.mixer.x_proj.weight for layer in model.backbone.layers],
        *[layer.mixer.dt_proj.weight for layer in model.backbone.layers],
        *[layer.mixer.dt_proj.bias for layer in model.backbone.layers],
        *[layer.mixer.out_proj.weight for layer in model.backbone.layers],
        *[layer.mixer.A_log for layer in model.backbone.layers],
        *[layer.mixer.D for layer in model.backbone.layers],
    ]
    shared_classifier = torch.equal(
        model.backbone.embedding.weight, model.lm_head.weight
    )
    if not shared_classifier:
        weights.append(model.lm_head.weight)
    for w in weights:
        assert (
            w.numel() % group_size == 0
        ), f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

    # write
    out_file = open(filepath, "wb")
    # first write out the header. the header will be 256 bytes
    # 2) write version, which will be int
    out_file.write(struct.pack("i", version))
    # 3) write the params, which will be 7 ints
    n_layers = model.config.n_layer
    vocab_size = model.config.vocab_size
    if vocab_size % model.config.pad_vocab_size_multiple != 0:
        vocab_size += (
            model.config.pad_vocab_size_multiple
            - vocab_size % model.config.pad_vocab_size_multiple
        )
    dim = model.config.d_model
    d_state = model.backbone.layers[0].mixer.d_state
    d_conv = model.backbone.layers[0].mixer.d_conv
    expand = model.backbone.layers[0].mixer.expand
    dt_rank = model.backbone.layers[0].mixer.dt_rank
    max_seq_len = MAX_SEQUENCE_LENGTH
    header = struct.pack(
        "iiiiiiii",
        n_layers,
        vocab_size,
        dim,
        d_state,
        d_conv,
        expand,
        dt_rank,
        max_seq_len,
    )
    out_file.write(header)
    # 4) write some other flags
    out_file.write(struct.pack("B", int(shared_classifier)))
    out_file.write(struct.pack("i", group_size))  # group size used for quantization
    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b"\0" * pad)
    # now that the header is done, let's write out the model

    # first let's write out all the params that we are keeping in fp32: the norms
    for layer in model.backbone.layers:  # attention norms
        serialize_fp32(out_file, layer.norm.weight)

    serialize_fp32(out_file, model.backbone.norm_f.weight)  # final pre-classifier norm

    del model
    gc.collect()

    # now let's write out all the params that we are quantizing to Q8_0
    # note we skip classifier weights, which are shared with the embedding
    ew = []
    weights_len = len(weights)
    i = 0
    while weights != []:
        w = weights.pop(0)
        q, s, err = quantize_q80(w, group_size)
        # save the int8 weights to file
        serialize_int8(out_file, q)  # save the tensor in int8
        serialize_fp32(out_file, s)  # save scale factors
        print(s.shape)
        # logging
        ew.append((err, w.shape))
        print(
            f"{i+1}/{weights_len} quantized {tuple(w.shape)} to Q8_0 with max error {err}"
        )

        del w
        gc.collect()

        i += 1

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


def export_mamba_for_c(model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    out_file = open(filepath, "wb")

    # 1) write the params, which will be 7 ints
    n_layers = model.config.n_layer
    vocab_size = model.config.vocab_size
    if vocab_size % model.config.pad_vocab_size_multiple != 0:
        vocab_size += (
            model.config.pad_vocab_size_multiple
            - vocab_size % model.config.pad_vocab_size_multiple
        )
    dim = model.config.d_model
    d_state = model.backbone.layers[0].mixer.d_state
    d_conv = model.backbone.layers[0].mixer.d_conv
    expand = model.backbone.layers[0].mixer.expand
    dt_rank = model.backbone.layers[0].mixer.dt_rank
    max_seq_len = MAX_SEQUENCE_LENGTH
    header = struct.pack(
        "iiiiiiii",
        n_layers,
        vocab_size,
        dim,
        d_state,
        d_conv,
        expand,
        dt_rank,
        max_seq_len,
    )
    out_file.write(header)

    # 2) write some other flags and padding
    shared_classifier = torch.equal(model.embedding.weight, model.lm_head.weight)
    out_file.write(struct.pack("B", int(shared_classifier)))
    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b"\0" * pad)

    # 3) now let's write out all the params
    weights = [
        model.backbone.embedding.weight,
        *[layer.norm.weight for layer in model.backbone.layers],
        *[layer.mixer.in_proj.weight for layer in model.backbone.layers],
        *[layer.mixer.conv1d.weight for layer in model.backbone.layers],
        *[layer.mixer.conv1d.bias for layer in model.backbone.layers],
        *[layer.mixer.x_proj.weight for layer in model.backbone.layers],
        *[layer.mixer.dt_proj.weight for layer in model.backbone.layers],
        *[layer.mixer.dt_proj.bias for layer in model.backbone.layers],
        *[layer.mixer.out_proj.weight for layer in model.backbone.layers],
        *[layer.mixer.A_log for layer in model.backbone.layers],
        *[layer.mixer.D for layer in model.backbone.layers],
        model.backbone.norm_f.weight,
    ]

    if not shared_classifier:
        weights.append(model.lm_head.weight)

    for w in weights:
        serialize_fp32(out_file, w)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


def model_export(checkpoint, file_path, quantized=0, group_size=64):
    if quantized == 1:
        model = MambaLMHeadModel.from_pretrained(
            checkpoint, dtype=torch.float32, device="cpu"
        )
        version2_export(model, file_path, group_size=group_size)
    else:
        model = MambaLMHeadModel.from_pretrained(
            "SalmanHabeeb/mamba-130m-1000", dtype=torch.float32, device="cpu"
        )
        export_mamba_for_c(model, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="huggingface model",
        default="SalmanHabeeb/mamba-130m-1000",
    )
    parser.add_argument(
        "--quantized",
        type=int,
        help="quantized or not",
        default=0,
    )
    parser.add_argument(
        "--group_size",
        type=int,
        help="quantization group size",
        default=64,
    )
    args = parser.parse_args()
    model_export(args.checkpoint, args.filepath, args.quantized, args.group_size)
