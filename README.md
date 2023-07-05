# color_web
Faiss + OpenCV + FastAPI + LMDB <br>
Uses color histograms (512 bin) <br>
Supported operations: add new image, delete image, find similar images by image file, find similar images by image id

```pip3 install -r requirements.txt```

```generate_rgb_histograms.py ./path_to_img_folder``` -> generates features  
```--use_int_filenames_as_id=0``` - images get sequential ids  
```--use_int_filenames_as_id=1``` - image id is parsed from filename ("123.jpg" -> 123)  
  
```add_to_index.py``` -> adds features from lmdb to Flat index  
  
```color_web.py``` -> web microservice  
```GET_FILENAMES=1 color_web.py``` -> when searching, include filename in search results  


DOCKER:  
build image - ```docker build -t qwertyforce/color_web:1.0.0 --network host -t qwertyforce/color_web:latest ./```  
  
run interactively - ```docker run -ti --rm -p 127.0.0.1:33335:33335 --network=ambience_net --mount type=bind,source="$(pwd)"/data,target=/app/data --name color_web qwertyforce/color_web:1.0.0```  
  
run as deamon - ```docker run -d --rm -p 127.0.0.1:33335:33335 --network=ambience_net --mount type=bind,source="$(pwd)"/data,target=/app/data --name color_web qwertyforce/color_web:1.0.0 ```  

