import numpy as np
from os import listdir
from joblib import Parallel, delayed
from tqdm import tqdm
import lmdb
import cv2
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str,nargs='?', default="./../test_images")
parser.add_argument('--use_int_filenames_as_id',choices=[0,1], type=int, default=0)
args = parser.parse_args()

IMAGE_PATH = args.image_path
USE_INT_FILENAMES = args.use_int_filenames_as_id

def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(4, 'big')
    
DB_filename_to_id = lmdb.open('./filename_to_id.lmdb',map_size=50*1_000_000) #50mb
DB_id_to_filename = lmdb.open('./id_to_filename.lmdb',map_size=50*1_000_000) #50mb
DB = lmdb.open('./rgb_histograms.lmdb',map_size=5000*1_000_000) #500mb

if USE_INT_FILENAMES == 0:
    with DB_id_to_filename.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            curs.last()
            x = curs.item()
            SEQUENTIAL_GLOBAL_ID = int_from_bytes(x[0]) # zeros if id_to_filename.lmdb is empty
    SEQUENTIAL_GLOBAL_ID+=1

def check_if_exists_by_file_name(file_name):
    if USE_INT_FILENAMES:
        image_id = int(file_name[:file_name.index('.')])
        image_id = int_to_bytes(image_id)
    else:
        with DB_filename_to_id.begin(buffers=True) as txn:
            image_id = txn.get(file_name.encode(), default=False)
            if not image_id:
                return False
    
    with DB.begin(buffers=True) as txn:
        x = txn.get(image_id, default=False)
        if x:
            return True
        return False

def get_features(query_image):
    query_hist_combined = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    query_hist_combined = query_hist_combined.flatten()
    query_hist_combined = query_hist_combined*10000000
    query_hist_combined = np.divide(query_hist_combined, query_image.shape[0]*query_image.shape[1], dtype=np.float32)
    return query_hist_combined

def read_img_file(f):
    img = Image.open(f)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def calc_hists(file_name):
    img_path = IMAGE_PATH+"/"+file_name
    try:
        query_image = np.array(read_img_file(img_path))
        image_features = get_features(query_image)
        return [file_name, image_features.tobytes()]
    except:
        print(f'error reading {img_path}')
        return None

file_names = listdir(IMAGE_PATH)
print(f"images in {IMAGE_PATH} = {len(file_names)}")

new_images = []
for file_name in tqdm(file_names):
    if check_if_exists_by_file_name(file_name):
        continue
    new_images.append(file_name)

print(f"new images = {len(new_images)}")
new_images = [new_images[i:i + 100000] for i in range(0, len(new_images), 100000)]
for batch in new_images:
    rgb_hists = Parallel(n_jobs=-1, verbose=1)(delayed(calc_hists)(file_name) for file_name in batch)
    rgb_hists = [i for i in rgb_hists if i]  # remove None's
    file_name_to_id = []
    id_to_file_name = []    
    for i in range(len(rgb_hists)):
        if USE_INT_FILENAMES:
            idx_of_dot = rgb_hists[i][0].index('.')
            image_id = int_to_bytes(int(rgb_hists[i][0][:idx_of_dot]))
        else:
            image_id = int_to_bytes(SEQUENTIAL_GLOBAL_ID)
            SEQUENTIAL_GLOBAL_ID+=1

        file_name = rgb_hists[i][0].encode()
        rgb_hists[i][0] = image_id
        rgb_hists[i] = tuple(rgb_hists[i])
        file_name_to_id.append((file_name, image_id))
        id_to_file_name.append((image_id, file_name))
        
    with DB_filename_to_id.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(file_name_to_id)

    with DB_id_to_filename.begin(write=True, buffers=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(id_to_file_name)

    print("pushing data to db")
    with DB.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(rgb_hists)
