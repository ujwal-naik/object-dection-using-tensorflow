"""
**1: Importing libraries**
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image,PIL.ImageFont,PIL.ImageDraw
import tensorflow as tf
import tensorflow_datasets as tfds

"""**2: visulization utilities**"""

im_width = 75
im_heigth = 75
use_normalized_coordinates = True

from matplotlib import use
def draw_bounding_boxes_on_image_array(image, boxes, color=['r', 'g', 'b'], thickness=1,display_str_list=(1)):
    image_pil = PIL.Image.fromarray(image)
    rgbimg = PIL.Image.new("RGBA", image_pil.size)
    rgbimg.paste(image_pil)
  
    draw_bounding_boxes_on_image(image, boxes, color=[], thickness=1,display_str_list=())
    return np.array(rgbimg)

def draw_bounding_boxes_on_image(image,boxes, color=[], thickness=1,display_str_list=()): 
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[-1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):

        draw_bounding_boxes_on_image(image , boxes[i,0],boxes[i,3],boxes[i,2],color[i],
                                     thickness,display_str_list[i])
def draw_bounding_boxes_on_image(image ,ymin, xmin, ymax, xmax,
                                 color='red',thickness=1,use_normalized_coordinates = True):
    draw = PIL.ImageDraw.Draw(image)

    image_actual_width, image_actual_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * image_actual_width, xmax * image_actual_width,
                                      ymin * image_actual_height, ymax * image_actual_height)
    else:
        (left, right, top, bottom) = (xmin, xmax,
                                      ymin, ymax)

    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
              (left, top),(left,top)], width=thickness, fill=color)

def dataset_to_numpy_util(training_dataset, validation_dataset, N):
    batch_train_ds = training_dataset.unbatch().batch(N)

    if tf.executing_eagerly():
        for validation_digits, (validation_labels , validation_bboxes) in validation_dataset:
            validation_digits = validation_digits.numpy()
            validation_labels = validation_labels.numpy()
            validation_bboxes = validation_bboxes.numpy()
            break
        for training_digits, (training_labels , training_bboxes) in validation_dataset:
            training_digits = training_digits.numpy()
            training_labels = training_labels.numpy()
            training_bboxes = training_bboxes.numpy()
            break

    validation_labels = np.argmax(validation_labels, axis=-1)
    training_labels = np.argmax(training_labels, axis=-1 )
    return (training_digits, training_labels, training_bboxes,
            validation_digits, validation_labels, validation_bboxes)

MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/data/fonts/ttf")
def create_digits_from_local_fonts(n):
  font_labels = []
  img = PIL.Image.new('LA',(75*n, 75),color = (0,255))
  font_1 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'DejaVuSansMono-Oblique.ttf'), 25)
  font_2 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'STIXGeneral.ttf'), 25)
  d = PIL.ImageDraw.Draw(img)
  for i in range(n):
    font_labels.append(i % 10)
    d.text((7+i*75,0 if i<10 else -4), str(i%10), fill = (255,255), font=font_1 if i<10 else font_2)
  font_didits = np.array(img.getdata(), np.float32)[:,0] / 255.0
  font_didits = np.reshape(np.stack(np.split(np.reshape(font_didits, (75, 75, n)), n, axis=2), axis= 0) [n , 75*75])
  return font_didits, font_labels

def display_digits_with_boxes(digitss, predictions, pred_bboxes, bboxes, iou, title):
    n = 10

    indexes = np.random.choice(len(predictions), size=n)
    n_digits = digitss[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]

    n_iou = []
    if len(iou) > 0:
        n_iou = iou[indexes]

    if len(pred_bboxes) > 0:
        n_pred_bboxes = pred_bboxes[indexes]

    if len (bboxes) > 0:
        n_bboxes = bboxes[indexes]

    n_digits = n_digits * 255.0
    n_digits = n_digits.reshape((n, 75, 75))
    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(10):
      ax = fig.add_subplot(1,10, i+1)
      bboxes_to_plot = []
      if (len(pred_bboxes) > i):
        bboxes_to_plot.extend(n_pred_bboxes[i])

      if (len(bboxes) > i):
        bboxes_to_plot.extend(n_bboxes[i])

      img_to_draw = draw_bounding_boxes_on_image_array(image=n_digits[i],
                                                       boxes=asarray(bboxes_to_plot),
                                                       color=['red','green'],
                                                       display_str_list=["True", 'pred'])
      plt.xlabel(n_predictions[i])
      plt.xticks([])
      plt.yticks([])

      if n_predictions[i] != n_labels[i]:
        ax.xaxis.label.set_color('red')

      plt.imshow(img_to_draw)

      if len(iou) > i:
        color = "black"
        if (n_iou[i][0] < iou_threshold):
          color = "red"
        
        ax.text(0.2, -0.3, "iou: %s" % (n_iou[i][0]), color=color, transform=ax.transAxes)
