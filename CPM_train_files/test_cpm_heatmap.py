import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from gen_data import gaussian_kernel
from google.colab.patches import cv2_imshow


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


# def draw_image(image, key_points):
# 	"""
# 	Draw limbs in the image.
# 	:param image: The test image.
# 	:param key_points: The key points of the person in the test image.
# 	:return: The painted image.
# 	"""
# 	'''ALl limbs of person:
# 	head top->neck
# 	neck->left shoulder
# 	left shoulder->left elbow
# 	left elbow->left wrist
# 	neck->right shoulder
# 	right shoulder->right elbow
# 	right elbow->right wrist
# 	neck->left hip
# 	left hip->left knee
# 	left knee->left ankle
# 	neck->right hip
# 	right hip->right knee
# 	right knee->right ankle
# 	'''
# 	# limbs = [[13, 12], [12, 9], [9, 10], [10, 11], [12, 8], [8, 7], [7, 6], [12, 3], [3, 4], [4, 5], [12, 2], [2, 1],
# 	#          [1, 0]]
	
# 	'''
# 	Monkey limbs:
# 	nose (0) -> left eye (1)
# 	nose (0) -> right eye (2)
# 	head (3) -> neck (4)
# 	neck (4) -> left shoulder (5)
# 	left shoulder (5) -> left elbow (6)
# 	left elbow (6) -> left wrist (7)
# 	neck (4) -> right shoulder (8)
# 	right shoulder (8) -> right elbow (9)
# 	right elbow (9) -> right wrist (10)
# 	neck (4) -> hip (11)
# 	hip (11) -> left knee (12)
# 	left knee (12) -> left ankle (13)
# 	hip (11) -> right knee (14)
# 	right knee (14) -> right ankle (15)
# 	hip (11) -> tail (16)
# 	'''
# 	limbs = [[0, 1], [0, 2], [3, 4], [4, 5], [5, 6], [6, 7], [4, 8], [8, 9], [9, 10], [4, 11], [11, 12], [12, 13],
# 	         [11, 14], [14,15], [11,16]]

# 	# draw key points
# 	for key_point in key_points:
# 		x = key_point[0]
# 		y = key_point[1]
# 		cv2.circle(image, (x, y), radius=1, thickness=-1, color=(0, 0, 255))

# 	# draw limbs
# 	for limb in limbs:
# 		start = key_points[limb[0]]
# 		end = key_points[limb[1]]
# 		color = (0, 0, 255)  # BGR
# 		cv2.line(image, tuple(start), tuple(end), color, thickness=1, lineType=4)

# 	return image

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


if __name__ == "__main__":
	model = torch.load('/content/drive/Shareddrives/S22_CSCI_5561_Project/Nate_CPM/Trained_model_2:26_morning/best_cpm.pth').cuda()

	image_path = '/content/heatmap_predictions/good_maps/test_0000100.jpg' # good prediction result
	# image_path = '/content/heatmap_predictions/bad_maps/test_0000027.jpg' # bad prediction result
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

	for i in range(6):
		heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, center_var)
		heatList = [heat1, heat2, heat3, heat4, heat5, heat6]
		key_points = get_key_points(heatList[i], height=height, width=width)

		image = draw_image(cv2.imread(image_path), key_points)

		# cv2_imshow(image)
		# cv2.waitKey(0)

		# cv2.imwrite(image_path.rsplit('.', 1)[0] + '_ans=.jpg', image)
		cv2.imwrite(image_path.rsplit('.', 1)[0] + f'heat{i+1}.jpg', image)
		cv2.imwrite('test_img.jpg', image)

