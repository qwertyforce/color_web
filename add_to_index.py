from tqdm import tqdm
import numpy as np
import lmdb
import faiss
from modules.byte_ops import int_from_bytes

DB_features = lmdb.open("./data/rgb_histograms.lmdb", readonly=True)
dim = 512
faiss_dim = dim
quantizer = faiss.IndexFlat(faiss_dim, faiss.METRIC_L1)
index = faiss.IndexIDMap2(quantizer)

def get_all_data_iterator(batch_size=10000):
    with DB_features.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            temp_ids = np.zeros(batch_size,np.int64)
            temp_features = np.zeros((batch_size,dim),np.float32)
            retrieved = 0
            for data in curs.iternext(keys=True, values=True):
                temp_ids[retrieved] = int_from_bytes(data[0])
                temp_features[retrieved] = np.frombuffer(data[1],dtype=np.float32)
                retrieved+=1
                if retrieved == batch_size:
                    retrieved=0
                    yield temp_ids, temp_features
            if retrieved != 0:
                yield temp_ids[:retrieved], temp_features[:retrieved]

for ids, features in tqdm(get_all_data_iterator(100000)):
    index.add_with_ids(features,ids)
faiss.write_index(index,"./data/populated.index")