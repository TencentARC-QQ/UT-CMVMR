import lmdb
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
#  input is the json file with tag embed(from extract_tag_embedding_top1.py)
input_json = sys.argv[1]
out_lmdb = sys.argv[2]

def main():
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    pair_f = open(input_json)
    pair = json.load(pair_f)
    pair_f.close()
    env = lmdb.open(out_lmdb)
    txn = env.begin(write=True)
    all_tag = []
    for vid, value in pair.items():
        tag = value[1]
        if tag not in all_tag:
            all_tag.append(tag)
    for tag in tqdm(all_tag):
        embeddings = model.encode(tag)
        embeddings = list(map(lambda x: x.tolist(), embeddings))
        embed = np.array(embeddings, dtype='float32')
        txn.put(tag.encode(), embed.tobytes())
    txn.commit()
    env.close()

if __name__ == "__main__":
    main()