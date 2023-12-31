import struct
import torch
from torch import nn
import argparse
from model import Mamba

MAX_SEQUENCE_LENGTH = 256

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def export_mamba_for_c(model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    out_file = open(filepath, 'wb')

    # 1) write the params, which will be 7 ints
    n_layers = model.args.n_layer
    vocab_size = model.args.vocab_size
    dim = model.args.d_model
    d_state = model.args.d_state
    d_conv = model.args.d_conv
    expand = model.args.expand
    dt_rank = model.args.dt_rank
    max_seq_len = MAX_SEQUENCE_LENGTH
    header = struct.pack('iiiiiiii', n_layers,
                         vocab_size, dim, d_state, d_conv, expand, dt_rank,
                         max_seq_len)
    out_file.write(header)

    # 2) write some other flags and padding
    shared_classifier = torch.equal(model.embedding.weight, model.lm_head.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # 3) now let's write out all the params
    weights = [
        model.embedding.weight,
        *[layer.norm.weight for layer in model.layers],
        *[layer.mixer.in_proj.weight for layer in model.layers],
        *[layer.mixer.conv1d.weight for layer in model.layers],
        *[layer.mixer.conv1d.bias for layer in model.layers],
        *[layer.mixer.x_proj.weight for layer in model.layers],
        *[layer.mixer.dt_proj.weight for layer in model.layers],
        *[layer.mixer.dt_proj.bias for layer in model.layers],
        *[layer.mixer.out_proj.weight for layer in model.layers],
        *[layer.mixer.A_log for layer in model.layers],
        *[layer.mixer.D for layer in model.layers],
        model.norm_f.weight,
    ]

    if not shared_classifier:
        weights.append(model.lm_head.weight)

    for w in weights:
        serialize_fp32(out_file, w)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

def model_export(checkpoint, file_path):
    model = Mamba.from_pretrained("state-spaces/mamba-130m")
    export_mamba_for_c(model, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--checkpoint", type=str, help="huggingface model", default="SalmanHabeeb/mamba-130m-1000")
    args = parser.parse_args()
    model_export(args.checkpoint, args.filepath)
