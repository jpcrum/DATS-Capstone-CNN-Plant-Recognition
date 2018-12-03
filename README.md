# DATS-Capstone-CNN-Plant-Recognition

Preprocessing (Preprocessing Jupyter Notebook Folder):

- All in Jupyter Notebooks
	-Reset all paths as needed
	
    Image_Augmentation:
	- Rotates and flips images to augment the image dataset 10-fold

    Image EDA:
	- Examines image sizes and creates testing and validation datasets 

    Image Preprocessing:
	- Examines effects of normalization, hue masking, and median blurring on plant dataset


Neural Networks (Networks Folder):
	Keras:
	- CNN.py	

	PyTorch:
	- MakeCSV.py: Run on training and testing image data before CNN. This will make a csv file
		     with image paths and labels
	- CNN_Pytorch.py: Link to newly created csv below line: "if __name__ == '__main__':" to load images 
	


Application (Application Folder):

	Import flask and flask-wtf:
	- Open anaconda terminal
	- conda install flask
	- conda install flask-wtf
	- set FLASK_APP=routes.py (no spaces around '=')
	- set FLASK_DEBUG=1
	- flask run

	Using App:
	- Open port that app is running on (prints when running)
	- Click "Upload Photos"
	- Copy and paste image folder path into input bar
	- Press "Analyze"

