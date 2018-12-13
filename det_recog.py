#-*- coding:utf-8 -*-
#'''
# Created on 18-10-19 下午4:08
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import config
import ctpn_model
from utils import TextProposalConnectorOriented
import utils
import cv2
import keras.backend as K
import ctc_model
from transform import four_point_transform

# this is my local checkpoint directory path, please replace it with your own
ctpn_weight_path = os.path.join('/home/gaolijun/workspace/keras_chinese_ocr/checkpoints', 'ep80_vgg16_loss0.0191.h5')
ctc_weight_path = os.path.join('/home/gaolijun/workspace/keras_chinese_ocr/checkpoints','densenet_ctc_ep10-0.095_0.102_0.9840_0.9839.h5')

width = 1024


img_path = os.path.join(config.base_dir, '001.png')
#####################  detection ######################
image = cv2.imread(img_path)
image = utils.resize(image, width=width)
image_ori = image.copy()
h, w, c = image.shape
image = image - config.IMAGE_MEAN
image = np.expand_dims(image, axis=0)

infer_model = ctpn_model.create_ctpn_model()
infer_model.load_weights(ctpn_weight_path)

cls, regr, cls_prob = infer_model.predict(image)

anchor = utils.gen_anchor((int(h/16), int(w/16)), 16)
bbox = utils.bbox_transfor_inv(anchor, regr)
bbox = utils.clip_box(bbox, [h, w])

fg = np.where(cls_prob[0,:,1]>0.7)[0]
select_anchor = bbox[fg, :]
select_score = cls_prob[0, fg, 1]
select_anchor = select_anchor.astype(np.int32)

keep_index = utils.filter_bbox(select_anchor, 16)

# nsm
select_anchor = select_anchor[keep_index]
select_score = select_score[keep_index]
select_score = np.reshape(select_score,(select_score.shape[0],1))
nmsbox = np.hstack((select_anchor,select_score))
keep = utils.nms(nmsbox,0.3)
select_anchor = select_anchor[keep]
select_score = select_score[keep]

#text line
textConn = TextProposalConnectorOriented()
text = textConn.get_text_lines(select_anchor,select_score,[h,w])

################### recognition ##########################
chars = utils.get_char()
ctc_infer_model = ctc_model.create_densenet_ctc_infer_model(img_h=config.img_h,
                                              nclass=len(chars)
                                              )
ctc_infer_model.load_weights(ctc_weight_path)


def pred_ctc(model, gray_image):
    image = utils.resize(gray_image, height=config.img_h)

    image = image.astype(np.float32) / 255 - 0.5
    image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)  # for example, (1, 32, 280, 1)

    pred = model.predict(image)
    # argmax = np.argmax(pred, axis=-1)[0]
    out = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1],)[0][0])[0]
    out = u''.join(utils.id_to_char(chars)[i] for i in out)

    return out


def get_sorted_bboxes(bboxes, thresh_height=20):
    text_sorted = np.array(sorted([t[:-1] for t in bboxes], key=lambda x: x[1]))
    bboxes_lines = []
    i = 0
    while i < text_sorted.shape[0]:
        cur_box = text_sorted[i]
        if i + 1 >= text_sorted.shape[0]:
            bboxes_lines.append([list(cur_box)])
            break
        line = [list(cur_box)]
        cur_h = thresh_height  # cur_box[5] - cur_box[1]
        # next_box = text_sorted[i + 1]
        for next_box in text_sorted[i+1:]:
            if next_box[1] - cur_box[1] < cur_h:
                line.append(list(next_box))
        line = sorted(line, key=lambda s: s[0])
        bboxes_lines.append(line)
        i += len(line)

    return bboxes_lines


image_c = image_ori.copy()
for line in get_sorted_bboxes(text):
    print('--'*10, len(line))
    out_line_texts = []
    for i in line:
        i = [int(j) for j in i]
        pts = np.array([(i[k], i[k + 1]) for k in range(0, 7, 2)])
        warped_image = four_point_transform(image_ori, pts)

        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        ctc_out = pred_ctc(ctc_infer_model, warped_image)
        out_line_texts.append(ctc_out)

        cv2.line(image_c,(i[0],i[1]),(i[2],i[3]),(0,0,128),2)
        cv2.line(image_c,(i[0],i[1]),(i[4],i[5]),(0,0,128),2)
        cv2.line(image_c,(i[6],i[7]),(i[2],i[3]),(0,0,128),2)
        cv2.line(image_c,(i[4],i[5]),(i[6],i[7]),(0,0,128),2)

    out_line_texts = ''.join(out_line_texts)
    print(out_line_texts)



