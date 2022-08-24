# color_web
Faiss + OpenCV + FastAPI + LMDB <br>
Uses color histograms (512 bin) <br>
Supported operations: add new image, delete image, find similar images by image file, find similar images by image id

```pip3 install -r requirements.txt```

```generate_rgb_histograms.py ./path_to_img_folder``` -> generates features  
```add_to_index.py``` -> adds features from lmdb to Flat index  
```color_web.py``` -> web microservice  
