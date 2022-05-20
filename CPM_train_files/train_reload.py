from gen_data import MONKEY_DATA
from Transformers import Compose, RandomCrop, RandomResized, TestResized
from utils import AverageMeter
from cpm import CPM
import torch.utils.data.dataloader
import torch.nn as nn
import pickle

if __name__ == "__main__":
	training_dataset_path = '/content/train_dataset/images/*.jpg'
	val_data_path = '/content/valid_dataset/images/*.jpg'
	model_save_path = '/content/model/cpm.pth'
	best_model_path = '/content/model/best_cpm.pth'
	train_landmarks = pickle.load(open('/content/train_dataset/train_landmarks.data', 'rb'))
	val_landmarks = pickle.load(open('/content/valid_dataset/val_landmarks.data', 'rb'))

	criterion = nn.MSELoss().cuda()

	# model = CPM(k=17).cuda()
	model = torch.load(model_save_path).cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	train_losses = AverageMeter()
	val_losses = AverageMeter()
	min_losses = 999.0

	torch.save(model, model_save_path)
	
	epoch = 0
	while epoch < 20:
		print('epoch ', epoch)
		"""--------Train--------"""
		# Training data
		data = MONKEY_DATA(train_landmarks, training_dataset_path, 8, Compose([RandomResized(), RandomCrop(368)]))
		print("Data created")
		train_loader = torch.utils.data.dataloader.DataLoader(data, batch_size=8)
		print("Data loaded")
		for j, data in enumerate(train_loader):
			inputs, heatmap, centermap = data

			inputs = inputs.cuda()
			heatmap = heatmap.cuda(non_blocking=True)
			centermap = centermap.cuda(non_blocking=True)

			input_var = torch.autograd.Variable(inputs)
			heatmap_var = torch.autograd.Variable(heatmap)
			centermap_var = torch.autograd.Variable(centermap)

			heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, centermap_var)

			loss1 = criterion(heat1, heatmap_var)
			loss2 = criterion(heat2, heatmap_var)
			loss3 = criterion(heat3, heatmap_var)
			loss4 = criterion(heat4, heatmap_var)
			loss5 = criterion(heat5, heatmap_var)
			loss6 = criterion(heat6, heatmap_var)

			loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
			train_losses.update(loss.item(), inputs.size(0))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('Train Loss: ', train_losses.avg)
		torch.save(model, model_save_path)

		'''--------Validation--------'''
		# Validation
		print('-----------Validation-----------')
		# Validation data
		data = MONKEY_DATA(val_landmarks, val_data_path, 8, Compose([TestResized(368)]))
		val_loader = torch.utils.data.dataloader.DataLoader(data, batch_size=8)

		model.eval()

		for j, data in enumerate(val_loader):
			inputs, heatmap, centermap = data

			inputs = inputs.cuda()
			heatmap = heatmap.cuda(non_blocking=True)
			centermap = centermap.cuda(non_blocking=True)

			input_var = torch.autograd.Variable(inputs)
			heatmap_var = torch.autograd.Variable(heatmap)
			centermap_var = torch.autograd.Variable(centermap)

			heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, centermap_var)

			loss1 = criterion(heat1, heatmap_var)
			loss2 = criterion(heat2, heatmap_var)
			loss3 = criterion(heat3, heatmap_var)
			loss4 = criterion(heat4, heatmap_var)
			loss5 = criterion(heat5, heatmap_var)
			loss6 = criterion(heat6, heatmap_var)

			loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
			val_losses.update(loss.item(), inputs.size(0))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print('Validation Loss: ', val_losses.avg)
		if val_losses.avg < min_losses:
			# Save best cpm
			torch.save(model, best_model_path)
			min_losses = val_losses.avg

		model.train()

		epoch += 1
