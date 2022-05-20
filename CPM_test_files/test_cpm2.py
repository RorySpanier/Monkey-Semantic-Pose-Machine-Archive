import os
import json
import pprint
import itertools
import argparse
import glob
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from gen_data import gaussian_kernel
from google.colab.patches import cv2_imshow

import matplotlib.pyplot as plt


def get_key_points(heatmap6, height, width):
	"""
	Get all key points from heatmap6.
	:param heatmap6: The heatmap6 of CPM cpm.
	:param height: The height of original image.
	:param width: The width of original image.
	:return: All key points of the person in the original image.
	"""
	# Get final heatmap
	heatmap = np.asarray(heatmap6.cpu().data)[0]

	key_points = []
	# Get k key points from heatmap6
	for i in heatmap[1:]:
		# Get the coordinate of key point in the heatmap (46, 46)
		y, x = np.unravel_index(np.argmax(i), i.shape)

		# Calculate the scale to fit original image
		scale_x = width / i.shape[1]
		scale_y = height / i.shape[0]
		x = int(x * scale_x)
		y = int(y * scale_y)

		key_points.append([x, y])

	return key_points


def draw_image(image, key_points):
  """
  Draw limbs in the image.
  :param image: The test image.
  :param key_points: The key points of the person in the test image.
  :return: The painted image.
  """

  """
  0: right eye
  1: left eye
  2: nose
  3: head
  4: neck
  5: right shoulder
  6: right elbow
  7: right wrist
  8: left shoulder
  9: left elbow
  10: left wrist
  11: hip
  12: right knee
  13: right ankle
  14: left knee
  15: left ankle
  16: tail
  """
  pink = (255, 204, 204)
  red = (0, 0, 255)
  blue = (255, 0, 0)
  orange = (102, 102, 255)
  light_blue = (255, 128, 0)
  cyan = (255, 255, 0)
  green = (0, 255, 0)
  yellow = (0, 255, 255)
  light_green = (178, 255, 102)


  limbs = [[0, 2], [1, 2], [3, 4], [4, 5], [4, 8], [4, 11], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [11, 14], [11, 16], [12, 13], [14, 15]]
  colors = [pink, pink, red, blue, blue, orange, light_blue, cyan, light_blue, cyan, green, green, yellow, light_green, light_green]

  # draw key points
  for key_point in key_points:
    x = key_point[0]
    y = key_point[1]
    cv2.circle(image, (x, y), radius=2, thickness=2, color=(255, 255, 255))

  # draw limbs
  for limb, color in zip(limbs, colors):
    start = key_points[limb[0]]
    end = key_points[limb[1]]
    cv2.line(image, tuple(start), tuple(end), color, thickness=2, lineType=4)

  return image
  
def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_path', type=str, dest='test_path', help='the path to all test images')
  parser.add_argument('--json_path', type=str, dest='json_path', help='the path to the json file')
  parser.add_argument('--model_path', type=str, dest='model_path', help='path to the trained model')
  parser.add_argument('--pred_path', type=str, dest='pred_path', help='path to the results')

  return parser.parse_args()


if __name__ == "__main__":
  args = parse()
  print(f"test_path:\t{args.test_path}\njson_path:\t{args.json_path}\nmodel_path:\t{args.model_path}\npred_path:\t{args.pred_path}\n")
  model = torch.load(args.model_path).cuda()

  image_path_list = sorted(glob.glob(args.test_path + "*.jpg"))
  print(f"Found {len(image_path_list)} validated image filenames")

  jf = json.load(open(args.json_path))['data']

  for i, image_path in tqdm(enumerate(image_path_list)):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image = np.asarray(image, dtype=np.float32)
    image = cv2.resize(image, (368, 368), interpolation=cv2.INTER_CUBIC)

    # Normalize
    image -= image.mean()
    image = F.to_tensor(image)

    # Generate center map
    centermap = np.zeros((368, 368, 1), dtype=np.float32)
    kernel = gaussian_kernel(size_h=368, size_w=368, center_x=184, center_y=184, sigma=3)
    kernel[kernel > 1] = 1
    kernel[kernel < 0.01] = 0
    centermap[:, :, 0] = kernel
    centermap = torch.from_numpy(np.transpose(centermap, (2, 0, 1)))

    image = torch.unsqueeze(image, 0).cuda()
    centermap = torch.unsqueeze(centermap, 0).cuda()

    model.eval()
    input_var = torch.autograd.Variable(image)
    center_var = torch.autograd.Variable(centermap)

    heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, center_var)
    key_points = get_key_points(heat6, height=height, width=width)

    image = draw_image(cv2.imread(image_path), key_points)
    cv2.imwrite(os.path.join(args.pred_path, f"prediction_{i:06d}.jpg"), image)

    jf[i]['landmarks'] = list(itertools.chain(*key_points))

  with open(args.json_path, 'w') as outfile:
    json.dump(jf, outfile, ensure_ascii=False, indent=4)

  print("Complete!")