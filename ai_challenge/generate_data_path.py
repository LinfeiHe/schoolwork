"""
generate data from source, implement only once
"""


import json
import os

root = os.getcwd()
data_path = os.path.join(root, 'data/')
train_path = data_path + 'train/'
val_path = data_path + 'val/'
test_path = data_path + 'test/'
src_tr_json = os.path.join(root, 'scene/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json')
src_val_json = os.path.join(root, 'scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json')
src_tr_path = os.path.join(root, 'scene/ai_challenger_scene_train_20170904/scene_train_images_20170904/')
src_val_path = os.path.join(root, 'scene/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/')
src_test_path = os.path.join(root, 'scene/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/')


def mkdir(path):
    path = path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
        return True


mkdir(data_path)
mkdir(train_path)
mkdir(val_path)
mkdir(test_path)



with open(src_tr_json, 'r') as f:
    tr_list_dict = json.loads(f.read())

with open(src_val_json, 'r') as f:
    val_list_dict = json.loads(f.read())

for i in range(len(tr_list_dict)):
    mkdir(train_path + tr_list_dict[i]['label_id'])
    src = src_tr_path + tr_list_dict[i]['image_id']
    dst = train_path + tr_list_dict[i]['label_id'] + '/' + tr_list_dict[i]['image_id']
    os.symlink(src, dst)

for i in range(len(val_list_dict)):
    mkdir(val_path + val_list_dict[i]['label_id'])
    src = src_val_path + val_list_dict[i]['image_id']
    dst = val_path + val_list_dict[i]['label_id'] + '/' + val_list_dict[i]['image_id']
    os.symlink(src, dst)

file_list = os.listdir(src_test_path)
for i in range(len(file_list)):
    src = src_test_path + file_list[i]
    dst = test_path + file_list[i]
    os.symlink(src, dst)

