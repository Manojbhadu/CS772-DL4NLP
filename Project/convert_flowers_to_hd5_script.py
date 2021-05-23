import os
from os.path import join, isfile
import numpy as np
import h5py
from glob import glob
from PIL import Image
import yaml
import io
import pdb


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-cls-token')

with open('config.yaml', 'r') as f:
	config = yaml.load(f)

images_path = config['flowers_images_path']
text_path = config['flowers_text_path']
datasetDir = config['flowers_dataset_path']

val_classes = open(config['flowers_val_split_path']).read().splitlines()
train_classes = open(config['flowers_train_split_path']).read().splitlines()
test_classes = open(config['flowers_test_split_path']).read().splitlines()

f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')

for _class in sorted(os.listdir(text_path)):
	d = os.path.join(text_path, _class)
	if os.path.isdir(d):
		print(_class)
		split = ''
		if _class in train_classes:
			split = train
		elif _class in val_classes:
			split = valid
		elif _class in test_classes:
			split = test
		
		txt_path = os.path.join(text_path, _class)
		for txt_file in sorted(glob(txt_path + "/*.txt")):
			image_name = txt_file.split("/")[-1][:-4]
			img_path = os.path.join(images_path,image_name)+".jpg"
			example_name = image_name 

			f = open(txt_file, "r")
			txt = f.readlines()
			f.close()
 
			txt = [i.rstrip("\n") for i in txt]

			img = open(img_path, 'rb').read()

			txt_choice = np.random.choice(range(10), 5)


			txt = np.array(txt)
			###Need to implement embedzss
			txt = txt[txt_choice]

			sentences = txt
			sentence_embeddings = model.encode(sentences)




			dt = h5py.special_dtype(vlen=str)

			for c, e in enumerate(sentence_embeddings):
			
				ex = split.create_group(example_name + '_' + str(c))
				ex.create_dataset('name', data=example_name)
				ex.create_dataset('img', data=np.void(img))
				ex.create_dataset('embeddings', data=e)
				ex.create_dataset('class', data=_class)
				ex.create_dataset('txt', data=txt[c].astype(object), dtype=dt)

			print(example_name, txt[1], _class)
			
	

