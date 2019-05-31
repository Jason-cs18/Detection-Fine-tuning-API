import numpy as np
import glob
import cv2
import os
import operator
import matplotlib.pyplot as plt
from collections import Counter
import random

def transform(bbs, img_width, img_height):
	out = []
	for bb in bbs:
		xmin, ymin, xmax, ymax = bb[:4]
		x_center = (xmax+xmin)/2/img_width
		y_center = (ymax+ymin)/2/img_height
		bb_width = (xmax-xmin)/img_width
		bb_height = (ymax-ymin)/img_height
		if bb[-1] == sorted_x[-1][0]:
			# index = 0, top-1
			out.append([0, x_center, y_center, bb_width, bb_height])
		elif bb[-1] == sorted_x[-2][0]:
			# index = 1, top-2
			out.append([1, x_center, y_center, bb_width, bb_height])
		else:
			# index = 2, top-3
			out.append([2, x_center, y_center, bb_width, bb_height])
	return out

current_directory = os.getcwd()
valid_labels = np.load('%s/valid_labels.npy'%current_directory, allow_pickle=True).item()

all_bbs = []
for key, value in valid_labels.items():
	all_bbs.extend(value)

class_nums = dict(Counter(np.array(all_bbs)[:, -1]))
sorted_x = sorted(class_nums.items(), key=operator.itemgetter(1))

sample_img = glob.glob('%s/images/*' % current_directory)[0]

height, width, channel = cv2.imread(sample_img).shape

train_imgs = random.sample(valid_labels.keys(), int(1.0*len(valid_labels)))

# generate labels
for img, bbs in valid_labels.items():
	current_bbs = transform(bbs, width, height)
	train_file = open("%s/labels/%s.txt" % (current_directory, img), "w")
	for bb in current_bbs:
		train_file.write("%s %s %s %s %s\n" % (str(bb[0]), str(bb[1]), str(bb[2]), str(bb[3]), str(bb[4]))) 
	train_file.close()

train_file = open("%s/train.txt" % current_directory, "w") 

for img in train_imgs:
	train_file.write('%s/images/' % current_directory+img+".png\n") 
train_file.close() 

train_file = open("%s/val.txt" % current_directory, "w") 

for img in train_imgs:
	train_file.write('%s/images/' % current_directory+img+".png\n") 
train_file.close()  

