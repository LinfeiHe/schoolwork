import numpy as np
import torch
import json
import os
import collections


MODE = 4  # best


def ensemble(inputs5, mode=1):
    res_densenet161 = inputs5['densenet161']
    res_nasnetalarge = inputs5['nasnetalarge']
    res_resnet152 = inputs5['resnet152']
    res_vgg19_bn = inputs5['vgg19_bn']
    res_inceptionv4 = inputs5['inceptionv4']
    top1_densenet161 = np.argsort(-res_densenet161, axis=1)[:, :1]
    top1_nasnetalarge = np.argsort(-res_nasnetalarge, axis=1)[:, :1]
    top1_resnet152 = np.argsort(-res_resnet152, axis=1)[:, :1]
    top1_vgg19_bn = np.argsort(-res_vgg19_bn, axis=1)[:, :1]
    top1_inceptionv4 = np.argsort(-res_inceptionv4, axis=1)[:, :1]
    top3_densenet161 = np.argsort(-res_densenet161, axis=1)[:, :3]
    top3_nasnetalarge = np.argsort(-res_nasnetalarge, axis=1)[:, :3]
    top3_resnet152 = np.argsort(-res_resnet152, axis=1)[:, :3]
    top3_vgg19_bn = np.argsort(-res_vgg19_bn, axis=1)[:, :3]
    top3_inceptionv4 = np.argsort(-res_inceptionv4, axis=1)[:, :3]
    top5_densenet161 = np.argsort(-res_densenet161, axis=1)[:, :5]
    top5_nasnetalarge = np.argsort(-res_nasnetalarge, axis=1)[:, :5]
    top5_resnet152 = np.argsort(-res_resnet152, axis=1)[:, :5]
    top5_vgg19_bn = np.argsort(-res_vgg19_bn, axis=1)[:, :5]
    top5_inceptionv4 = np.argsort(-res_inceptionv4, axis=1)[:, :5]

    rows = res_densenet161.shape[0]
    if mode == 1:
        temp = res_nasnetalarge + res_resnet152 + res_densenet161 + res_vgg19_bn + res_inceptionv4
        top3 = np.argsort(-temp, axis=1)
        outputs = top3[:, :3]
    elif mode == 2:
        temp = 0.2 * res_densenet161 + 0.4 * res_nasnetalarge + 0.2 * res_resnet152 + 0.1 * res_vgg19_bn + 0.1 * res_inceptionv4
        top3 = np.argsort(-temp, axis=1)
        outputs = top3[:, :3]
    elif mode == 3:
        items = []
        for row in range(rows):
            temp = list(top1_densenet161[row]) + list(top1_nasnetalarge[row]) \
                   + list(top1_resnet152[row]) + list(top1_vgg19_bn[row]) + list(top1_inceptionv4[row])
            top3 = collections.Counter(temp).most_common(3)
            if len(top3) == 1:
                temp_1 = list(top3_densenet161[row]) + list(top3_nasnetalarge[row]) \
                       + list(top3_resnet152[row]) + list(top3_vgg19_bn[row]) + list(top3_inceptionv4[row])
                while True:
                    if top3[0][0] in temp_1:
                        temp_1.remove(top3[0][0])
                    else:
                        break
                top3_1 = collections.Counter(temp_1).most_common(3)
                items.append(np.array([[top3[0][0], top3_1[0][0], top3_1[1][0]]]))
            elif len(top3) == 2:
                temp_1 = list(top3_densenet161[row]) + list(top3_nasnetalarge[row]) \
                         + list(top3_resnet152[row]) + list(top3_vgg19_bn[row]) + list(top3_inceptionv4[row])
                while True:
                    if top3[0][0] in temp_1:
                        temp_1.remove(top3[0][0])
                    elif top3[1][0] in temp_1:
                        temp_1.remove(top3[1][0])
                    else:
                        break
                top3_1 = collections.Counter(temp_1).most_common(3)
                items.append(np.array([[top3[0][0], top3[1][0], top3_1[0][0]]]))
            else:
                items.append(np.array([[top3[0][0], top3[1][0], top3[2][0]]]))
        outputs = np.concatenate(items)
    elif mode == 4:
        items = []
        for row in range(rows):
            temp = list(top3_densenet161[row]) + list(top3_nasnetalarge[row]) \
                   + list(top3_resnet152[row]) + list(top3_vgg19_bn[row]) + list(top3_inceptionv4[row])
            top3 = collections.Counter(temp).most_common(3)
            items.append(np.array([[top3[0][0], top3[1][0], top3[2][0]]]))
        outputs = np.concatenate(items)
    elif mode == 5:
        items = []
        for row in range(rows):
            temp = list(top5_densenet161[row]) + list(top5_nasnetalarge[row]) \
                   + list(top5_resnet152[row]) + list(top5_vgg19_bn[row]) + list(top5_inceptionv4[row])
            top3 = collections.Counter(temp).most_common(3)
            items.append(np.array([[top3[0][0], top3[1][0], top3[2][0]]]))
        outputs = np.concatenate(items)
    return outputs


def generate_json(inputs):
    out2real = {}
    out = [str(x) for x in range(80)]
    real = sorted(out)
    for idx, item in enumerate(real):
        out2real[idx] = item            # 分类的文件夹以str排序，而不是数字顺序



    cids = torch.load('val_outputs.pkl_label')
    image_filenames = cids['resnet152']
    temp = []
    for rows in range(inputs.shape[0]):
        id = [int(out2real[inputs[rows][0]]), int(out2real[inputs[rows][1]]), int(out2real[inputs[rows][2]])]
        item = {"image_id": image_filenames[rows],
                "label_id": id}
        temp.append(item)
    with open('submit.json', 'w') as json_file:
        print(len(temp))
        json_file.write(json.dumps(temp))
    print('json generate')


def _eval(ref, submit):
    cwd = os.getcwd()
    py = os.path.join(cwd, 'scene_eval.py')
    submit = os.path.join(cwd, submit)
    command = 'python "{}" --submit "{}" --ref "{}"'.format(py, submit, ref)
    os.system(command)


def val_eval():
    val_outputs = torch.load('val_outputs.pkl')
    val_inceptionv4 = np.load('val_inceptionv4.npy')
    val_normalized = val_outputs
    val_normalized['inceptionv4'] = val_inceptionv4

    em = ensemble(val_normalized, MODE)
    generate_json(em)
    ref = '/home/helinfei/PycharmProjects/ai_challenge/scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
    submit = 'submit.json'
    _eval(ref, submit)


def generate_test_json():
    test_outputs = torch.load('test_outputs.pkl')
    test_inceptionv4 = np.load('test_inceptionv4.npy')
    test_normalized = test_outputs
    test_normalized['inceptionv4'] = test_inceptionv4
    em = ensemble(test_normalized, MODE)
    generate_json(em)





val_eval()
generate_test_json()