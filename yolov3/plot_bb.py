import numpy as np
import glob
import operator
import matplotlib.pyplot as plt
from collections import Counter

def parse_data_cfg(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

classes = load_classes(parse_data_cfg('data/coco.data')['names'])

all_imgs = glob.glob('./video/images/*')

output_labels = np.load('./video/output_labels.npy', allow_pickle=True).item()

all_bbs = []
for key, value in output_labels.items():
    all_bbs.extend(value)

plt.hist(np.array(all_bbs)[:, 4], normed=True, cumulative=True, label='CDF',
         histtype='step', alpha=0.8, color='b', linewidth=1.5)
plt.ylabel('Percentage', fontsize=20)
plt.xlabel('Confidence', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('CDF for pseudo-bounding-boxes', fontsize=12)
plt.savefig('./video/percentage.png')

class_nums = dict(Counter(np.array(all_bbs)[:, -1]))
sorted_x = sorted(class_nums.items(), key=operator.itemgetter(1))
print('-*'*10)
print('There are %d valid images: ' % len(output_labels))
print("There are %d images that don't have bbs." % (len(all_imgs)-len(output_labels)))
print("There are %d objects in the video." % len(Counter(np.array(all_bbs)[:, -1])))
print('-*'*10)
print('We only fine-tune model on top-3 classes!')
print('Use these index to update ./yolov3/data/custom/custom.names')
print('Index-0: ', classes[int(sorted_x[-1][0])])
print('Index-1: ', classes[int(sorted_x[-2][0])])
print('Index-2: ', classes[int(sorted_x[-3][0])])
print('-*'*10)

valid_index = [sorted_x[-1][0], sorted_x[-2][0], sorted_x[-3][0]]

valid_labels = {}
for key, values in output_labels.items():
    current_img = []
    for bb in values:
        if bb[-1] in valid_index:
            current_img.append(bb)
    if len(current_img) > 0:
        valid_labels[key] = current_img
np.save('./video/valid_labels.npy', valid_labels)
name_list = [classes[int(index)] for index in class_nums.keys()]
num_list = [value for index, value in class_nums.items()]
error = np.random.rand(len(num_list))
y_pos = np.arange(len(num_list))
plt.barh(y_pos, num_list, xerr=error, align='center', tick_label=name_list)
plt.xlabel('# bb', fontsize=15)
plt.ylabel('Class-Name', fontsize=15)
plt.savefig('./video/video_analysis.png')
