# -*- coding:utf-8 -*-
# '''
# Created on 18-12-12 上午9:59
#
# @Author: Greg Gao(laygin)
# '''
import os
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


def det_recog(image, infer_model, ctc_infer_model, width=1024):
    '''

    :param image: image object
    :param width: if not None, then resize image width, keep ratio
    :return:recognized texts
    '''
    if width is not None:
        image = utils.resize(image, width=width)
    image_ori = image.copy()

    def _detect(image):
        '''

        :param image: numpy array image, h,w,c
        :return: text location,a list which contains a list of x, y coors, prob. with shape (#lines, 9)
        '''
        h, w, c = image.shape
        image = image - config.IMAGE_MEAN
        image = np.expand_dims(image, axis=0)  # batch_sz, h, w, c

        _, regr, cls_prob = infer_model.predict(image)

        anchor = utils.gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = utils.bbox_transfor_inv(anchor, regr)
        bbox = utils.clip_box(bbox, [h, w])

        fg = np.where(cls_prob[0, :, 1] > 0.7)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)

        keep_index = utils.filter_bbox(select_anchor, 16)

        # nsm
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = utils.nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        # text line
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])
        return text

    def _recog(lines, image):
        '''

        :param lines: text coordinates from detection model
        :param image: a numpy array image with shape (h, w, c=3)
        :return: recognized text
        '''

        def __pred_ctc(model, gray_image):
            image = utils.resize(gray_image, height=32)

            image = image.astype(np.float32) / 255 - 0.5
            image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)  # for example, (1, 32, 280, 1)

            pred = model.predict(image)
            out = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], )[0][0])[0]
            out = u''.join(utils.id_to_char(chars)[i] for i in out)

            return out

        def __get_sorted_bboxes(bboxes, thresh_height=20):
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
                for next_box in text_sorted[i + 1:]:
                    if next_box[1] - cur_box[1] < cur_h:
                        line.append(list(next_box))
                line = sorted(line, key=lambda s: s[0])
                bboxes_lines.append(line)
                i += len(line)

            return bboxes_lines

        chars = utils.get_char()
        texts = []
        for line in __get_sorted_bboxes(lines):
            out_line_texts = []
            for i in line:
                i = [int(j) for j in i]
                pts = np.array([(i[k], i[k + 1]) for k in range(0, 7, 2)])
                warped_image = four_point_transform(image, pts)

                warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
                ctc_out = __pred_ctc(ctc_infer_model, warped_image)
                out_line_texts.append(ctc_out)
            out_line_texts = ' '.join(out_line_texts)
            # print(out_line_texts)

            texts.append(out_line_texts)

        return texts

    lines = _detect(image)
    texts = _recog(lines, image_ori)  # a list

    return texts


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # this is my local checkpoint directory
    ctpn_weight_path = os.path.join('./checkpoints',
                                    'ep80_vgg16_loss0.0191.h5')
    ctc_weight_path = os.path.join('./checkpoints',
                                   'densenet_ctc_ep12-0.176_0.152_0.9922_0.9909.h5')

    infer_model = ctpn_model.create_ctpn_model()
    infer_model.load_weights(ctpn_weight_path)

    chars = utils.get_char()
    ctc_infer_model = ctc_model.create_densenet_ctc_infer_model(img_h=32,
                                                                nclass=len(chars)
                                                                )
    ctc_infer_model.load_weights(ctc_weight_path)
    width = 1024
    img_path = os.path.join(config.base_dir, '001.png')

    print(det_recog(cv2.imread(img_path), infer_model=infer_model, ctc_infer_model=ctc_infer_model, width=width))

    pass
