import numpy as np
import torch
import math
import pickle
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
try:
	import umap.umap_ as umap
except:
	print('no umap installed, proceeding...')

def vae_loss(x_hat, x, mu, std, beta):
	"""
	Regular MSE based VAE loss with reconstruction and normalization terms
	"""
	reconstruction_loss = F.mse_loss(x_hat.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
	normalization_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum()
	return reconstruction_loss + beta * normalization_loss

def weighted_vae_loss(x_hat, x, mu, std, beta, weights):
	"""
	Weighted VAE loss for GMMs
	"""
	reconstruction_loss = F.mse_loss(x_hat.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
	normalization_loss = 0
	for w in range(len(weights)):
		weight = weights[0,0,0].detach()
		mu_of_w = mu[:, w, :]
		std_of_w = std[:, w, :]
		normalization_loss += weight * \
				-0.5 * (1 + 2 * std_of_w.log() - mu_of_w.pow(2) - std_of_w.pow(2)).sum()
	return reconstruction_loss + beta * normalization_loss

def double_vae_loss(x_hat, x, mu1, std1, mu2, std2, beta):
	"""
	Regular MSE based VAE loss with reconstruction and normalization terms
	"""
	reconstruction_loss = F.mse_loss(x_hat.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
	nr1_loss = -0.5 * (1 + 2 * std1.log() - mu1.pow(2) - std1.pow(2)).sum()
	nr2_loss = -0.5 * (1 + 2 * std2.log() - mu2.pow(2) - std2.pow(2)).sum()
	return reconstruction_loss + beta * (nr1_loss + nr2_loss)

def display(images, length, titles=[], size=(18, 6)):
	f, axes = plt.subplots(1, length, sharex='col', sharey='row', figsize=size)
	for i, title in enumerate(titles):
		axes[i].set_title(title)
	for i, image in enumerate(images):
		image = image.data.cpu().view(28, 28)
		if len(images) > 1:
			_ = axes[i].imshow(image, cmap='gray')
		else:
			_ = axes.imshow(image, cmap='gray')

def reparametrize(mu, std, device):
	"""
	Performs reparameterization trick z = mu + epsilon * std
	Where epsilon~N(0,1)
	"""
	mu = mu.expand(1, *mu.size())
	std = std.expand(1, *std.size())
	eps = torch.normal(0, 1, size=std.size()).to(device)
	return mu + eps * std

def get_dicts():
	keys = [
			'test_izx', 'test_izy', 'test_acc', 'test_error',
			'test_izx_ema', 'test_izy_ema', 'test_acc_ema', 'test_error_ema',
			'test_class_loss', 'test_info_loss', 'test_total_loss',
			'test_class_loss_ema', 'test_info_loss_ema', 'test_total_loss_ema',
			'train_izx', 'train_izy', 'train_acc', 'train_error', 'train_izx_ema',
			'train_izy_ema', 'train_acc_ema', 'train_error_ema',
			'train_class_loss', 'train_info_loss', 'train_total_loss',
			'k_size', 'num_weights', 'z_size', 'num_stochastic',
			'description', 'num_params'
			]
	book_keeper = {key: [] for key in keys}
	return book_keeper

def register_data(book_keeper, is_train, epoch_izx_bound, epoch_izy_bound,
				 epoch_accuracy, epoch_class_loss,
				 epoch_info_loss, epoch_total_loss):
	if is_train:
		book_keeper['train_izx'].append(epoch_izx_bound)
		book_keeper['train_izy'].append(epoch_izy_bound)
		book_keeper['train_acc'].append(epoch_accuracy)
		book_keeper['train_error'].append(1 - epoch_accuracy)
		book_keeper['train_class_loss'].append(epoch_class_loss)
		book_keeper['train_info_loss'].append(epoch_info_loss)
		book_keeper['train_total_loss'].append(epoch_total_loss)
	else:
		book_keeper['test_izx'].append(epoch_izx_bound)
		book_keeper['test_izy'].append(epoch_izy_bound)
		book_keeper['test_acc'].append(epoch_accuracy)
		book_keeper['test_error'].append(1 - epoch_accuracy)
		book_keeper['test_class_loss'].append(epoch_class_loss)
		book_keeper['test_info_loss'].append(epoch_info_loss)
		book_keeper['test_total_loss'].append(epoch_total_loss)
	return book_keeper

def get_n_params(model):
	pp=0
	for p in list(model.parameters()):
		nn=1
		for s in list(p.size()):
			nn = nn*s
		pp += nn
	return pp

def pickle_model_stats(model, pickle_save_path):
	with open(pickle_save_path, 'wb') as f:
		pickle.dump((model.train_loss, model.test_loss), f)

def plot_casted_embeddings(model, train_loader, device,
						  fmnist_classes, title, method,
						  take_stochastic=1):
	latent_list = []
	label_list = []
	label_text_list = []
	model.eval()
	for idx, (image_batch, label_batch) in enumerate(train_loader):
		with torch.no_grad():
			if model.num_weights > 1:
				(mu, _), _, _ = model(image_batch.to(device))
				mu = mu.view(train_loader.batch_size * model.num_weights, model.k)
				for label in label_batch:
					for w in range(model.num_weights):
						label_list.append(int(label))
						label_text_list.append(fmnist_classes[int(label)])
			elif model.num_stochastic > 1:
				if take_stochastic == 1:
					(mu, _), (_, _), _ = model(image_batch.to(device))
				elif take_stochastic == 2:
					(_, _), (mu, _), _ = model(image_batch.to(device))
				label_list += [int(label) for label in label_batch]
				label_text_list += [fmnist_classes[int(label)] for label in label_batch]
			else:
				(mu, _), _ = model(image_batch.to(device))
				label_list += [int(label) for label in label_batch]
				label_text_list += [fmnist_classes[int(label)] for label in label_batch]
			latent_list += [mu1.unsqueeze(0).cpu() for mu1 in mu]
		if idx > 100:
			break
	latent_array = np.array([tensor.cpu().numpy()[0] for tensor in latent_list])
	
	if method.lower() == 'tsne':
		E = TSNE(n_components=2).fit_transform(latent_array)
		f, a = plt.subplots(figsize=(15, 4))
		s = a.scatter(E[:,0], E[:,1], c=label_list, cmap='tab10')
		a.grid(False)
		a.axis('equal')
		a.set_title(title, fontsize=16)
		f.colorbar(s, ax=a, ticks=np.arange(10), boundaries=np.arange(11) - .5)
		return E

	elif method.lower() == 'umap':
		reducer = umap.UMAP()
		scaler = StandardScaler().partial_fit(latent_array)
		scaled_data = scaler.transform(latent_array)
		embedding = reducer.fit_transform(scaled_data)

		labels = pd.Series(label_text_list)
		fig, axes = plt.subplots(1, 1, figsize=(7, 7))
		c = [[c for c in range(1,11)][x] for x in labels\
			   .map({label: i for i, label in enumerate(labels.unique())})]

		scatter = axes.scatter(
							embedding[:, 0],
							embedding[:, 1],
							c=c,
							cmap='tab20c')

		# TODO: fix legend real class correlation
		legend1 = axes.legend(scatter.legend_elements()[0],\
							[text for text in fmnist_classes],\
							fontsize=16, prop={'size': 12})

		axes.add_artist(legend1)
		fig.canvas.draw()
		axes.set_title(title, fontsize=16)
		return reducer
	else:
		raise

class EMA_smoothning(object):
	"""
	Performs exponential moving average smoothing on model updates as per
	Polyak & Juditsky, 1992.
	This will be used as a second network refference when evaluating.
	"""
	def __init__(self, model, state_dict, beta_decay=0.999):
		self.model = model
		self.model.load_state_dict(state_dict, strict=True)
		self.beta_decay = beta_decay

	def update(self, new_state_dict):
		state_dict = self.model.state_dict()
		for key in state_dict.keys():
			state_dict[key] = (self.beta_decay) * state_dict[key] + (1 - self.beta_decay) * new_state_dict[key]
		self.model.load_state_dict(state_dict)

def loop_data(model, model_ema, dataloader, book_keeper,
			  is_train, model_save_path, beta, epochs,
			  device, optimizer=None, scheduler=None):
	"""
	loops over the dataset, collects metrics and train 
	a model if is_train is True
	"""
	if is_train:
		model.train()
		model_ema.model.train()
	else:
		model.eval()
		model_ema.model.eval()
		epochs = 1

	for e in range(epochs):
		epoch_class_loss = 0
		epoch_info_loss = 0
		epoch_izx_bound = 0
		epoch_izy_bound = 0
		epoch_total_loss = 0
		epoch_correct = 0
		epoch_samples = 0
		
		for batch_num, (images, labels) in enumerate(dataloader):
			x = images.to(device)
			y = labels.to(device)
			epoch_samples += y.size(0)
			if (model.num_weights > 1):
				(mu, std), logit, weights = model(x)
				batch_info_loss = 0
				for w in range(model.num_weights):
					weight = weights[0,0,0].detach()
					mu_of_w = mu[:, w, :]
					std_of_w = std[:, w, :]
					batch_info_loss += weight * (-0.5 * (1 + 2 * std_of_w.log() - mu_of_w.pow(2) - std_of_w.pow(2)).sum(1).mean().div(math.log(2)))
			elif model.num_stochastic > 1:
				(mu1, std1), (mu2, std2), logit = model(x)
				batch_info_loss1 = -0.5 * (1 + 2 * std1.log() - mu1.pow(2) - std1.pow(2)).sum(1).mean().div(math.log(2))
				batch_info_loss2 = -0.5 * (1 + 2 * std2.log() - mu2.pow(2) - std2.pow(2)).sum(1).mean().div(math.log(2))
				batch_info_loss = batch_info_loss1 + batch_info_loss2
			else:    
				(mu, std), logit = model(x) 
				batch_info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
			
			batch_class_loss = F.cross_entropy(logit, y).div(math.log(2))
			batch_total_loss = batch_class_loss + beta * batch_info_loss
			batch_izy_bound = math.log(10, 2) - batch_class_loss
			batch_izx_bound = batch_info_loss

			if is_train:
				optimizer.zero_grad()
				batch_total_loss.backward()
				optimizer.step()
				model_ema.update(model.state_dict())

			batch_prediction = F.softmax(logit, dim=1).max(1)[1]
			batch_accuracy = torch.eq(batch_prediction, y).float().mean()

			epoch_class_loss += batch_class_loss.data.item()
			epoch_info_loss += batch_info_loss.data.item()
			epoch_izx_bound += batch_izx_bound.data.item()
			epoch_izy_bound += batch_izy_bound.data.item()
			epoch_total_loss += batch_total_loss.data.item()
			epoch_correct += torch.eq(batch_prediction, y).float().sum()

			if is_train and (batch_num % 100) == 0:
				print('i:{} IZY:{:.2f} IZX:{:.2f}'.format(batch_num+1, batch_izy_bound.data.item(), batch_izx_bound.data.item()), end=' ')
				print('acc:{:.4f}'.format(batch_accuracy.data.item()), end=' ')
				print('err:{:.4f}'.format(1-batch_accuracy.data.item()))
		
		epoch_class_loss /= batch_num
		epoch_info_loss /= batch_num
		epoch_izx_bound /= batch_num
		epoch_izy_bound /= batch_num
		epoch_total_loss /= batch_num
		epoch_accuracy = epoch_correct.data.item() / epoch_samples

		book_keeper = register_data(book_keeper, is_train, epoch_izx_bound, epoch_izy_bound,
									 epoch_accuracy, epoch_class_loss,
									 epoch_info_loss, epoch_total_loss)

		if len(book_keeper['test_acc']) > 0 and not is_train:
			if (max(book_keeper['test_acc']) < epoch_accuracy):
				torch.save(model, model_save_path)
				print('Saved model to {}'.format(model_save_path))

		if not is_train:
			print('[TEST RESULTS]')
			print('IZY:{:.2f} IZX:{:.2f}'.format(epoch_izy_bound, epoch_izx_bound), end=' ')
			print('acc:{:.4f}'.format(epoch_accuracy), end=' ')
			print('err:{:.4f}'.format(1 - epoch_accuracy))

	return model, model_ema, book_keeper

def loop_data_vae(model, dataloader, beta,
				  is_train, model_save_path, epochs,
				  device, optimizer=None, scheduler=None):
	"""
	loops over the dataset, collects metrics and train 
	a model if is_train is True
	This is for a non IB VAE (regular VAE)
	"""
	if is_train:
		model.train()
	else:
		model.eval()
		epochs = 1

	for e in range(epochs):
		epoch_loss = 0        
		for batch_num, (images, labels) in enumerate(dataloader):
			x = images.to(device)
			y = labels.to(device)
			if model.num_weights > 1:
				(mu, std), decoded, weights = model(x)
				batch_loss = weighted_vae_loss(decoded.view(-1, 28*28), 
										x.view(-1, 28*28), mu, std, beta, weights)
			elif model.num_stochastic > 1:
				(mu1, std1), (mu2, std2), decoded = model(x)
				batch_loss = double_vae_loss(decoded.view(-1, 28*28), 
										x.view(-1, 28*28), mu1, std1, mu2, std2, beta)
			else:
				(mu, std), decoded = model(x)
				batch_loss = vae_loss(decoded.view(-1, 28*28), x.view(-1, 28*28),
									  mu, std, beta)
			if is_train:
				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()
			
			epoch_loss += batch_loss.item()

			if is_train and (batch_num % 100) == 0:
				print('batch loss:{:.2f}'.format(batch_loss.item()))
		
		epoch_loss /= batch_num
		if is_train:
			model.train_loss.append(epoch_loss)
		else:
			model.test_loss.append(epoch_loss)
		
		if len(model.test_loss) > 1 and not is_train:
			if (epoch_loss <= min(model.test_loss)):
				torch.save(model, model_save_path)
				print('Saved model to {}'.format(model_save_path))

		if not is_train:
			print('TEST RESULTS: epoch loss:{:.2f}'.format(epoch_loss))
	return model


def train_model(epochs, model, model_ema, train_loader, test_loader,
				device, lr, model_save_path, ema_save_path, pickle_save_path, beta):
	
	optimizer = optim.Adam(model.parameters(), lr, betas=(0.5,0.999))
	scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

	book_keeper = get_dicts()

	for epoch in range(epochs):
		print('[epoch {}]'.format(epoch))
		model, model_ema, book_keeper = loop_data(model, model_ema, train_loader, book_keeper,
												is_train=True, model_save_path=model_save_path,
												 beta=beta, epochs=1, device=device,
												optimizer=optimizer, scheduler=scheduler)

		model, model_ema, book_keeper = loop_data(model, model_ema, test_loader, book_keeper,
												is_train=False, model_save_path=model_save_path,
												 beta=beta, epochs=1, device=device,
												optimizer=optimizer, scheduler=scheduler)

		if (epoch % 2) == 0 and (epoch != 0):
			scheduler.step()


	print("----- Training complete -----")
	book_keeper['k_size'] = model.k
	book_keeper['z_size'] = model.decoder[-1].in_features
	book_keeper['num_params'] = int(get_n_params(model))
	book_keeper['description'] = model.description
	if hasattr(model, 'num_weights'):
		book_keeper['num_weights'] = model.num_weights
	else:
		book_keeper['num_weights'] = 1
	if hasattr(model, 'num_stochastic'):
		book_keeper['num_stochastic'] = model.num_stochastic
	else:
		book_keeper['num_stochastic'] = 1
	
	with open(pickle_save_path, 'wb') as f:
		pickle.dump(book_keeper, f)
	print('saved stats file to {}'.format(pickle_save_path))
	return book_keeper

def train_model_vae(epochs, model, train_loader, test_loader,
					device, lr, model_save_path, beta):
	"""
	Trains a model on a regular VAE task (not IB)
	"""
	optimizer = optim.Adam(model.parameters(), lr, betas=(0.5,0.999))
	scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

	for epoch in range(epochs):
		print('[epoch {}]'.format(epoch))
		model = loop_data_vae(model, train_loader, beta, is_train=True, 
				  model_save_path=model_save_path, epochs=1, device=device,
				  optimizer=optimizer, scheduler=scheduler)

		model = loop_data_vae(model, test_loader, beta, is_train=False, 
				  model_save_path=model_save_path, epochs=1, device=device,
				  optimizer=optimizer, scheduler=scheduler)

		if (epoch % 2) == 0 and (epoch != 0):
			scheduler.step()

	print("----- Training complete -----")
	return model
