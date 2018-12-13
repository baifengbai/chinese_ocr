#-*- coding:utf-8 -*-
#'''
# Created on 18-10-19 下午4:08
#
# @Author: Greg Gao(laygin)
#'''
import os


checkpoints_dir = './checkpoints'
base_dir = './images'

IMAGE_MEAN = [123.68, 116.779, 103.939]

img_h = 32
char_path = os.path.join(base_dir, 'char_std_5990.txt')