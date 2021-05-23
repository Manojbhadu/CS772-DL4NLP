# Text-to-Image-Synthesis 

The code for this project is pytorch. 

# Libraries used for this project 

1. torch
2. PIL
3. sentence-transformers
4. visdom
5. pdb
6. os
7. numpy
8. yaml
9. glob
10. h5py
11. os
12. io
13. tkinter

# Setting up the data directory 

1. Download images dataset from link : https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.
After downloading the dataset, extact the file and place "jpg/" folder inside the "Data/" directory.

2. Download the caption for the images from link : https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view
After downloading the captions tar file, extract the file and place files "testclasses.txt" , "trainclasses.txt", "valclasses.txt" and folder "text_c10" inside the "Data/" directory.

# Running the scripts 

1. First run the script "convert_flowers_to_hd5_script.py" using command line : python3 convert_flowers_to_hd5_script.py
to generate file "flowers.hdf5" in "Data/" directory. The "flowers.hdf5" contains the sentence embedding for text caption mapped to corresponding image.

2. Train the model using python script : python3 runtime.py --epochs 300
The generator and discrimator weights after each epoch are stored in folder "Data/checkpoints/"

3. For testing run the script "testing.py" using command line : python3 testing.py 
The GUI will be generated. Enter the description of flower. The clicking the button, the image of flower will be generated.