#Visualizations
Use the code here to visualize our data.


To get started, edit the `ROHIT_BASE_PATH` variable in `../run.py` and/or `init_paths.m`
to point to the directory that contains our data. i.e. the parent directory 
that the directory for each scan.

```
  ROHIT_BASE_PATH = /path/to/dataset/
```


Replace `Home_01_1` with whichever scene you wish to view 
###Visualize our images and bounding boxes, and virtually move around each scene.
#####Python
Run from the parent direction. (where the `run.py` file is)
  ```
    python run.py vis_boxes_and_move Home_01_1 
  ``` 
#####MATLAB
  ```
    >>>vis_boxes_and_move('Home_01_1')
  ``` 


###Visualize the camera positions and directions in each scene.
#####Python
  ```
    python run.py vis_camera_pos_dirs Home_01_1 
  ``` 
#####MATLAB
  ```
    >>>vis_camera_pos_dirs('Home_01_1')
  ``` 




