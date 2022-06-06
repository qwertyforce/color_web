import uvicorn
if __name__ == '__main__':
    uvicorn.run('color_web:app', host='127.0.0.1', port=33335, log_level="info")

from pydantic import BaseModel
from typing import Optional, Union
from fastapi import FastAPI, File, Form, HTTPException, Response, status
import faiss
import asyncio
from os.path import exists
import numpy as np
import cv2


import lmdb
DB = lmdb.open('./rgb_histograms.lmdb',map_size=5000*1_000_000) #5000mb

DATA_CHANGED_SINCE_LAST_SAVE = False
index = None

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

def delete_descriptor_by_id(id):
    with DB.begin(write=True,buffers=True) as txn:
        txn.delete(int_to_bytes(id))   #True = deleted False = not found


def add_descriptor(id, rgb_hist):
    with DB.begin(write=True, buffers=True) as txn:
        txn.put(int_to_bytes(id), np.frombuffer(rgb_hist, dtype=np.float32))

def init_index():
    global index
    if exists("./populated.index"):
        index = faiss.read_index("./populated.index")
    else:
        print("Index is not found! Exiting...")
        exit()


def read_img_file(image_data):
    return np.fromstring(image_data, np.uint8)
    
def get_features(image_buffer):
    query_image = cv2.cvtColor(cv2.imdecode(read_img_file(image_buffer), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    query_hist_combined = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    query_hist_combined = query_hist_combined.flatten()
    query_hist_combined = query_hist_combined*10000000
    query_hist_combined = np.divide(query_hist_combined, query_image.shape[0]*query_image.shape[1], dtype=np.float32)
    return query_hist_combined

def hist_similarity_search(target_features, k, distance_threshold):
    if k is not None:
        D, I = index.search(target_features, k)
        D = D.flatten()
        I = I.flatten()
    elif distance_threshold is not None:
        start_k=min(index.ntotal, 100)
        while True:
            D, I = index.search(target_features,start_k)
            D = D.flatten()
            I = I.flatten()
            if max(D) < distance_threshold:
                if(start_k == index.ntotal):
                    break
                start_k*=2
            else:
                indexes=np.where(D < distance_threshold)[0]
                D=D[indexes]
                I=I[indexes]
                break
            if(start_k > index.ntotal):
                    break
            print(start_k)

    res=[{"image_id":int(I[i]),"distance":float(D[i])} for i in range(len(D))]

    return res


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


class Item_image_id(BaseModel):
    image_id: int
    k: Union[str,int,None] = None
    distance_threshold: Union[str,float,None] = None

@app.post("/color_get_similar_images_by_id")
async def color_get_similar_images_by_id_handler(item: Item_image_id):
    try:
        k=item.k
        distance_threshold=item.distance_threshold
        if item.k:
            k = int(k)
        if item.distance_threshold:
            distance_threshold = float(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features = index.reconstruct(item.image_id).reshape(1,-1)
        similar = hist_similarity_search(target_features, k, distance_threshold)
        return similar
    except:
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
        results = hist_similarity_search(target_features, k, distance_threshold)
        return results
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Error in color_get_similar_images_by_image_buffer")


@app.post("/calculate_color_features")
async def calculate_color_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        features = get_features(image)
        add_descriptor(int(image_id), features)
        index.add_with_ids(features.reshape(1,-1), np.int64([image_id]))
        DATA_CHANGED_SINCE_LAST_SAVE = True
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=500, detail="Can't calculate color features")

@app.post("/delete_color_features")
async def delete_color_features_handler(item: Item_image_id):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        delete_descriptor_by_id(item.image_id)
        res = index.remove_ids(np.int64([item.image_id]))
        if res != 0: 
            DATA_CHANGED_SINCE_LAST_SAVE = True
        else: #nothing to delete
            print(f"err: no image with id {item.image_id}")
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=500, detail="Can't delete color features")

print(__name__)

def periodically_save_index(loop):
    global DATA_CHANGED_SINCE_LAST_SAVE, index
    if DATA_CHANGED_SINCE_LAST_SAVE:
        DATA_CHANGED_SINCE_LAST_SAVE=False
        faiss.write_index(index, "./populated.index")
    loop.call_later(10, periodically_save_index,loop)

if __name__ == 'color_web':
    init_index()
    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)
