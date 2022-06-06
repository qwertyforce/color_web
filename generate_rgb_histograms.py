import cv2
import numpy as np
from os import listdir
from joblib import Parallel, delayed
from tqdm import tqdm
import lmdb

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

DB = lmdb.open('./rgb_histograms.lmdb',map_size=5000*1_000_000) #5000mb
IMAGE_PATH = "./../images"


def check_if_exists_by_id(id):
    with DB.begin(buffers=True) as txn:
        x = txn.get(int_to_bytes(id),default=False)
        if x:
            return True
        return False

def get_features(query_image):
    query_hist_combined = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    query_hist_combined = query_hist_combined.flatten()
    query_hist_combined = query_hist_combined*10000000
    query_hist_combined = np.divide(query_hist_combined, query_image.shape[0]*query_image.shape[1], dtype=np.float32)
    return query_hist_combined

def calc_hists(file_name):
    file_id = int(file_name[:file_name.index('.')])
    img_path = IMAGE_PATH+"/"+file_name
    query_image = cv2.imread(img_path)
    if query_image is None:
        print(f'error reading {img_path}')
        return None
    if query_image.shape[2] == 1:
        query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
    else:
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    image_features = get_features(query_image)
    return (int_to_bytes(file_id), image_features.tobytes())


file_names = listdir(IMAGE_PATH)
file_names.sort()
print(f"images in {IMAGE_PATH} = {len(file_names)}")
new_images = []

for file_name in tqdm(file_names):
    file_id = int(file_name[:file_name.index('.')])
    if check_if_exists_by_id(file_id):
        continue
    new_images.append(file_name)

print(f"new images = {len(new_images)}")
new_images = [new_images[i:i + 100000] for i in range(0, len(new_images), 100000)]
for batch in new_images:
    rgb_hists = Parallel(n_jobs=-1, verbose=1)(delayed(calc_hists)(file_name) for file_name in batch)
    rgb_hists = [i for i in rgb_hists if i]  # remove None's
    print("pushing data to db")
    with DB.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(rgb_hists)
