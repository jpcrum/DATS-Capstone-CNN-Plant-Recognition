# DATS-Capstone-CNN-Plant-Recognition

Preprocessing:

- All in Jupyter Notebooks
	-Reset all paths as needed
	
    Image_Augmentation:
	- Rotates and flips images to augment the image dataset 10-fold

    Image EDA:
	-  

    Image Preprocessing:
	- Examines effects of normalization, hue masking, and median blurring on plant dataset



Application:

Import flask and flask-wtf:
- Open anaconda terminal
- conda install flask
- conda install flask-wtf

Run App:
- cd *Application folder*
- set FLASK_APP=routes.py (no spaces around '=')
- set FLASK_DEBUG=1
- flask run

Using App:
- Open port that app is running on (prints when running)
- Click "Upload Photos"
- Copy and paste image folder path into input bar
- Press "Analyze"

