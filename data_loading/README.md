#DATA LOADING

Here we provide two classes for working with our data in python.
See the `demo.py` file for a few different ways to use the class with/without pytorch.
The code is meant to be read, and so there are extensive(too many) comments.

###`active_vision_dataset.py`

Python class for loading our data.



###`active_vision_dataset_pytorch.py`

Very similar to `active_vision_dataset.py` but 
with a change in `__get_item__` function to work
with pytorch's DataLoader. This change is _NOT CORRECT_, 
but allows the code to be run with no errors. It causes a 
single bounding box to be output for each image, when some images
have multiple boxes or no boxes.

This file is just a placeholder for now.


