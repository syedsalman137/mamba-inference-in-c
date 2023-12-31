from transformers import AutoTokenizer
import argparse
import os
import struct
import tempfile

def get_merge_data(tokenizer):
   merge_data = None
   with tempfile.TemporaryDirectory() as tempdir_path:
       tokenizer.save_vocabulary(tempdir_path)

       with open(os.path.join(tempdir_path, "merges.txt"), "r") as f:
           # Split by lines, skip first line, which contains version info
           merge_data = f.read().splitlines()[1:]

   # Remove spaces in merge data, to merge tokens
   return [to_merge.replace(" ", "") for to_merge in merge_data]

def export_tokenizer_for_c(tokenizer, filepath):
  merge_data = get_merge_data(tokenizer)
  tokens, scores = [], []
  for i in range(tokenizer.vocab_size):

      # decode the token and light postprocessing
      t = tokenizer._tokenizer.model.id_to_token(i)
      s = None
      try:
        s = tokenizer.vocab_size - merge_data.index(t)
      except:
        s = -1
      if i == tokenizer.bos_token_id:
          t = '\n<s>\n'
      t = t.replace('Ġ', ' ')  # gptneoxtokenizer uses this character as whitespace
      t = t.replace('č', '\r') # gptneoxtokenizer uses this character as \r
      t = t.replace('Ĉ', '\b') # gptneoxtokenizer uses this character as \b
      t = t.replace('Ċ', '\n') # gptneoxtokenizer uses this character as \n
      t = t.replace('ĉ', '\t') # gptneoxtokenizer uses this character as \t

      b = t.encode('utf-8') # bytes of this token, utf-8 encoded

      tokens.append(b)
      scores.append(s)

  max_token_length = max(len(t) for t in tokens)

  # # write to a binary file
  with open(filepath, 'wb') as f:
      f.write(struct.pack("I", tokenizer.vocab_size))
      f.write(struct.pack("I", max_token_length))
      for token_bytes, score in zip(tokens, scores):
          f.write(struct.pack("fI", score, len(token_bytes)))
          f.write(token_bytes)
  
  print(f"Wrote tokenizer in {filepath}")

def tokenizer_export(tokenizer_name, filepath):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  tokenizer.eos_token = "<|endoftext|>"
  tokenizer.pad_token = tokenizer.eos_token
  export_tokenizer_for_c(tokenizer, filepath)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("filepath", type=str, help="output file path")
  parser.add_argument("--tokenizer", type=str, help="Huggingface path to tokenizer", default="EleutherAI/gpt-neox-20b")
  args = parser.parse_args()
  tokenizer_export(args.tokenizer, args.filepath)
