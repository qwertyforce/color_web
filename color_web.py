import uvicorn
if __name__ == '__main__':
    uvicorn.run('color_web:app', host='0.0.0.0', port=33335, log_level="info")
    exit()

from os import environ
if not "GET_FILENAMES" in environ:
    print("GET_FILENAMES not found! Defaulting to 0...")
    GET_FILENAMES = 0
else:
    if environ["GET_FILENAMES"] not in ["0","1"]:
        print("GET_FILENAMES has wrong argument! Defaulting to 0...")
        GET_FILENAMES = 0
    else:
        GET_FILENAMES = int(environ["GET_FILENAMES"])

import traceback
from pydantic import BaseModel
from typing import Optional, Union
from fastapi import FastAPI, File, Form, HTTPException, Response, status
import faiss
import asyncio
from os.path import exists
import numpy as np
import io 
from PIL import Image

from modules.lmdb_ops import get_dbs
from modules.byte_ops import int_to_bytes, int_from_bytes
from modules.color_ops import get_color_features

DATA_CHANGED_SINCE_LAST_SAVE = False
INDEX = None
app = FastAPI()

def main():
    global DB_rgb_hists, DB_filename_to_id, DB_id_to_filename
    init_index()
    DB_rgb_hists, DB_filename_to_id, DB_id_to_filename = get_dbs()

    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(4, 'big')

def check_if_exists_by_image_id(image_id):
    with DB_rgb_hists.begin(buffers=True) as txn:
        x = txn.get(int_to_bytes(image_id), default=False)
        if x:
            return True
        return False

def get_filenames_bulk(image_ids):
    image_ids_bytes = [int_to_bytes(x) for x in image_ids]

    with DB_id_to_filename.begin(buffers=False) as txn:
        with txn.cursor() as curs:
            file_names = curs.getmulti(image_ids_bytes)
    for i in range(len(file_names)):
        file_names[i] = file_names[i][1].decode()

    return file_names

def get_image_id_by_filename(file_name):
    with DB_filename_to_id.begin(buffers=True) as txn:
        image_id = txn.get(file_name.encode(), default=False)
        if not image_id:
            return False
        return int_from_bytes(image_id)
    
def delete_descriptor_by_id(image_id):
    image_id_bytes = int_to_bytes(image_id)
    with DB_rgb_hists.begin(write=True, buffers=True) as txn:
        txn.delete(image_id_bytes)   #True = deleted False = not found

    with DB_id_to_filename.begin(write=True, buffers=True) as txn:
        file_name_bytes = txn.get(image_id_bytes, default=False)
        txn.delete(image_id_bytes)  

    with DB_filename_to_id.begin(write=True, buffers=True) as txn:
        txn.delete(file_name_bytes) 

def add_descriptor(image_id, rgb_hist):
    file_name_bytes = f"{image_id}.online".encode()
    image_id_bytes = int_to_bytes(image_id)
    with DB_rgb_hists.begin(write=True, buffers=True) as txn:
        txn.put(image_id_bytes, rgb_hist.tobytes())

    with DB_id_to_filename.begin(write=True, buffers=True) as txn:
        txn.put(image_id_bytes, file_name_bytes)

    with DB_filename_to_id.begin(write=True, buffers=True) as txn:
        txn.put(file_name_bytes, image_id_bytes)

def read_img_file(image_buffer):
    img = Image.open(io.BytesIO(image_buffer))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
    
def get_features(image_buffer):
    query_image = np.array(read_img_file(image_buffer))
    query_hist_combined = get_color_features(query_image)
    return query_hist_combined

def hist_similarity_search(target_features, k, distance_threshold):
    if k is not None:
        D, I = INDEX.search(target_features, k)
        D = D.flatten()
        I = I.flatten()
    elif distance_threshold is not None: #range_search in faiss doesn't work with L1 distance
        start_k=min(INDEX.ntotal, 100)
        while True:
            D, I = INDEX.search(target_features,start_k)
            D = D.flatten()
            I = I.flatten()
            if max(D) < distance_threshold:
                if(start_k == INDEX.ntotal):
                    break
                start_k*=2
            else:
                indexes=np.where(D < distance_threshold)[0]
                D=D[indexes]
                I=I[indexes]
                break
            if(start_k > INDEX.ntotal):
                break
            print(start_k)
    res=[{"image_id":int(I[i]),"distance":float(D[i])} for i in range(len(D))]
    return res


@app.get("/")
async def read_root():
    return {"Hello": "World"}

class Item_delete_color_features(BaseModel):
    image_id: Union[int ,None] = None
    file_name: Union[None,str] = None

class Item_color_get_similar_images_by_id(BaseModel):
    image_id: int
    k: Union[str,int,None] = None
    distance_threshold: Union[str,float,None] = None

@app.post("/color_get_similar_images_by_id")
async def color_get_similar_images_by_id_handler(item: Item_color_get_similar_images_by_id):
    try:
        k=item.k
        distance_threshold=item.distance_threshold
        if item.k:
            k = int(k)
        if item.distance_threshold:
            distance_threshold = float(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features = INDEX.reconstruct(item.image_id).reshape(1,-1)
        similar = hist_similarity_search(target_features, k, distance_threshold)
        if GET_FILENAMES:
            file_names = get_filenames_bulk([el["image_id"] for el in similar])
            for i in range(len(similar)):
                similar[i]["file_name"] = file_names[i]
        return similar
    except:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="Error in color_get_similar_images_by_id")


@app.post("/color_get_similar_images_by_image_buffer")
async def color_get_similar_images_by_image_buffer_handler(image: bytes = File(...), k: Optional[str] = Form(None), distance_threshold: Optional[str] = Form(None)):
    try:
        if k:
            k = int(k)
        if distance_threshold:
            distance_threshold = float(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features=get_features(image).reshape(1,-1)
        similar = hist_similarity_search(target_features, k, distance_threshold)
        if GET_FILENAMES:
            file_names = get_filenames_bulk([el["image_id"] for el in similar])
            for i in range(len(similar)):
                similar[i]["file_name"] = file_names[i]
        return similar
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error in color_get_similar_images_by_image_buffer")


@app.post("/calculate_color_features")
async def calculate_color_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        image_id = int(image_id)
        if check_if_exists_by_image_id(image_id):
            return Response(content="Image with the same id is already in the db", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, media_type="text/plain")

        features = get_features(image)
        add_descriptor(image_id, features)
        INDEX.add_with_ids(features.reshape(1,-1), np.int64([image_id]))
        DATA_CHANGED_SINCE_LAST_SAVE = True
        return Response(status_code=status.HTTP_200_OK)
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Can't calculate color features")

@app.post("/delete_color_features")
async def delete_color_features_handler(item: Item_delete_color_features):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        if item.file_name:
            image_id = get_image_id_by_filename(item.file_name)
        else:
            image_id = item.image_id
        res = INDEX.remove_ids(np.int64([image_id]))
        if res != 0: 
            delete_descriptor_by_id(image_id)
            DATA_CHANGED_SINCE_LAST_SAVE = True
        else: #nothing to delete
            print(f"err: no image with id {image_id}")    
        return Response(status_code=status.HTTP_200_OK)
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Can't delete color features")

def periodically_save_index(loop):
    global DATA_CHANGED_SINCE_LAST_SAVE, INDEX
    if DATA_CHANGED_SINCE_LAST_SAVE:
        DATA_CHANGED_SINCE_LAST_SAVE=False
        faiss.write_index(INDEX, "./data/populated.index")
    loop.call_later(10, periodically_save_index,loop)

def init_index():
    global INDEX
    if exists("./data/populated.index"):
        INDEX = faiss.read_index("./data/populated.index")
    else:
        print("Index is not found! Exiting...")
        print("Creating empty index")
        import subprocess
        subprocess.call(['python3', 'add_to_index.py'])
        subprocess.call(['python', 'add_to_index.py']) #one should exist
        init_index()

main()