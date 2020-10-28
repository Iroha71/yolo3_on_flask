from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import os
import cv2
from absl import logging, app

from yolov3.yolov3_tf2.models import YoloV3
from yolov3.yolov3_tf2.dataset import transform_images
from yolov3.yolov3_tf2.utils import draw_outputs

def main(_argv):
# モデルの読み込み
  CLASS_NUM = 80
  WEIGHT_FILE = 'yolov3/checkpoints/yolov3.tf'
  yolo_model = YoloV3(classes=CLASS_NUM)
  yolo_model.load_weights(WEIGHT_FILE).expect_partial()

  # クラス名の読み込み
  CLASS_FILE = 'yolov3/data/coco.names'
  class_names = [c.strip() for c in open(CLASS_FILE).readlines()]
  print("== 検知開始 ==")
  # 画像読み込み
  IMAGE_FILE = 'yolov3/data/girl.png'
  img_raw = tf.image.decode_image(
    open(IMAGE_FILE, 'rb').read(), channels=3
  )
  img = tf.expand_dims(img_raw, 0)
  IMAGE_SIZE = 416
  img = transform_images(img, IMAGE_SIZE)
  boxes, scores, classes, nums = yolo_model(img)
  for i in  range(nums[0]):
    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                      np.array(scores[0][i]),
                                      np.array(boxes[0][i])))
  img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
  img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
  IMAGE_OUTPUT = './static/output.jpg'
  cv2.imwrite(IMAGE_OUTPUT, img)
  logging.info('output saved to: {}'.format(IMAGE_OUTPUT))
  
if __name__ == '__main__':
  app.run(main)