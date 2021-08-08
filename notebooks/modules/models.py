import numpy as np
import torch
import math
import os
import pickle
import modules.utils as utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torchvision.datasets import FashionMNIST

class MNIST_2STOCHASTIC(nn.Module):
	"""
	Implements the paper's IB NN with 2 stochastic layers
	"""
	def __init__(self, k, device):
		super(MNIST_2STOCHASTIC, self).__init__()
		self.k = k
		self.num_weights = 1
		self.num_stochastic = 2
		self.device = device
		self.description = '2 sequential stochastic layers'

		self.encoder = nn.Sequential(
			nn.Linear(784, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 2 * self.k))

		self.decoder = nn.Sequential(
				nn.Linear(self.k//2, 10))

		# Xavier initialization
		for _, module in self._modules.items():
			for layer in module:
				if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
							nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
							layer.bias.data.zero_()

	def forward(self, x):
		# squiwsh from shape (100,1,28,28) to (100,784)
		x = x.view(x.size(0),-1)
		z1_params = self.encoder(x)
		
		mu1 = z1_params[:, :self.k]
		std1 = F.softplus(z1_params[:, self.k:] - 5, beta=1)
		z1 = utils.reparametrize(mu1, std1, self.device).squeeze(0)

		mu2 = z1[:, :self.k//2]
		std2 = F.softplus(z1[:, self.k//2:] - 5, beta=1)
		z2 = utils.reparametrize(mu2, std2, self.device)

		logit = self.decoder(z2)
		return (mu1, std1), (mu2, std2), logit[0]

class MNIST_2STOCHASTIC_EXTENDED(nn.Module):
	"""
	Implements the paper's IB NN with 2 stochastic layers
	"""
	def __init__(self, k, device):
		super(MNIST_2STOCHASTIC_EXTENDED, self).__init__()
		self.k = k
		self.num_weights = 1
		self.num_stochastic = 2
		self.device = device
		self.description = '2 sequential stochastic layers with an intermediate linear layer'

		self.encoder = nn.Sequential(
			nn.Linear(784, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 2 * self.k))

		self.intermediate = nn.Linear(self.k, self.k * 2)

		self.decoder = nn.Sequential(
				nn.Linear(self.k, 10))

		# Xavier initialization
		for _, module in self._modules.items():
			if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
							nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
							module.bias.data.zero_()
			else:
				for layer in module:
					if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
								nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
								layer.bias.data.zero_()

	def forward(self, x):
		# squiwsh from shape (100,1,28,28) to (100,784)
		x = x.view(x.size(0),-1)
		z1_params = self.encoder(x)
		
		mu1 = z1_params[:, :self.k]
		std1 = F.softplus(z1_params[:, self.k:] - 5, beta=1)
		z1 = utils.reparametrize(mu1, std1, self.device).squeeze(0)

		z2_params = self.intermediate(z1)

		mu2 = z2_params[:, :self.k]
		std2 = F.softplus(z2_params[:, self.k:] - 5, beta=1)
		z2 = utils.reparametrize(mu2, std2, self.device)

		logit = self.decoder(z2)
		return (mu1, std1), (mu2, std2), logit[0]


class MixtureModle(nn.Module):
	"""
	NN module that implements a weighted mixture model while learning the weights
	"""
	def __init__(self, num_weights: int) -> None:
		super(MixtureModle, self).__init__()
		self.num_weights = num_weights
		# uniform initialization
		self.weights = Parameter(torch.ones(1, num_weights) / num_weights)
		self.register_parameter('bias', None)
		
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		weights = torch.softmax(self.weights, dim=1) # force weights to sum to 1
		# not using diagonal matrix as grad cannot be computed
		# TODO: consider reducing the redundent first dimension (also form reparam function)
		weights = torch.stack(torch.tensor_split(self.weights,self.num_weights,dim=1), dim=1).repeat(1,1,input.shape[-1])
		weighted = input * weights
		weighted = weighted.sum(dim=2)
		return weighted, weights

	def extra_repr(self) -> str:
		return 'weights={}'.format(self.weights)

class MNIST_IB_VAE_GMM(nn.Module):
	"""
	Implements the paper's IB NN with a GMM
	"""
	def __init__(self, k, w, device):
		super(MNIST_IB_VAE_GMM, self).__init__()
		self.k = k
		self.w = w
		assert k % w == 0
		self.part_len = k // w
		self.num_weights = w
		self.num_stochastic = 1
		self.device = device
		self.description = 'GMM IB VAE with num_weights gaussians'

		self.encoder = nn.Sequential(
			nn.Linear(784, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 2 * self.k * self.w))
		self.gmm = MixtureModle(w)
		self.decoder = nn.Sequential(
				nn.Linear(self.k, 10))
		# Xavier initialization
		for name, module in self._modules.items():
			if name == 'gmm':
				continue
			for layer in module:
				if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
							nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
							layer.bias.data.zero_()

	def forward(self, x):
		# squiwsh from shape (100,1,28,28) to (100,784)
		x = x.view(x.size(0),-1)
		z_params = self.encoder(x)
		
		mu = torch.stack(torch.split(z_params[:, :(self.k * self.w)], self.w, dim=-1), dim=2)
		# softplus transformation (soft relu) and a -5 bias is added as in the paper
		std = F.softplus(z_params[:, self.k * self.w:] - 5, beta=1)
		std = torch.stack(torch.split(std, self.w, dim=-1), dim=2)
		z = utils.reparametrize(mu, std, self.device)
		z, weights = self.gmm(z)
		logit = self.decoder(z)
		return (mu, std), logit[0], weights

class MNIST_VANILA_IB_VAE(nn.Module):
	"""
	Direct implementation of the paper's MNIST net
	Only one shot eval (no MC) - work well for beta <= 1e-3
	"""
	def __init__(self, k, device):
		super(MNIST_VANILA_IB_VAE, self).__init__()
		self.k = k
		self.num_weights = 1
		self.num_stochastic = 1
		self.device = device
		self.description = 'Vanilla IB VAE as per the paper'

		self.encoder = nn.Sequential(
			nn.Linear(784, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 2 * self.k))

		self.decoder = nn.Sequential(
				nn.Linear(self.k, 10))

		# Xavier initialization
		for _, module in self._modules.items():
			for layer in module:
				if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
							nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
							layer.bias.data.zero_()

	def forward(self, x):
		# squiwsh from shape (100,1,28,28) to (100,784)
		x = x.view(x.size(0),-1)
		z_params = self.encoder(x)
		mu = z_params[:, :self.k]
		# softplus transformation (soft relu) and a -5 bias is added as in the paper
		std = F.softplus(z_params[:, self.k:] - 5, beta=1)
		z = utils.reparametrize(mu, std, self.device)
		logit = self.decoder(z)
		return (mu, std), logit[0]

class FMNIST_VANILA_VAE(nn.Module):
	"""
	Just your regular variational auto encoder (not IB)
	"""
	def __init__(self, k, device):
		super(FMNIST_VANILA_VAE, self).__init__()
		self.device = device
		self.description = 'Vanilla VAE'
		self.k = k
		self.num_weights = 1
		self.num_stochastic = 1
		self.train_loss = []
		self.test_loss = []

		self.encoder = nn.Sequential(
			nn.Linear(784, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 2 * self.k))

		self.decoder = nn.Sequential(
				nn.Linear(self.k, 1024),
				nn.ReLU(True),
				nn.Linear(1024, 28*28),
				nn.Sigmoid())

		# Xavier initialization
		for _, module in self._modules.items():
			for layer in module:
				if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
							nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
							layer.bias.data.zero_()

	def forward(self, x):
		# squiwsh from shape (100,1,28,28) to (100,784)
		x = x.view(x.size(0),-1)
		z_params = self.encoder(x)
		mu = z_params[:, :self.k]
		# softplus transformation (soft relu) and a -5 bias is added as in the paper
		std = F.softplus(z_params[:, self.k:] - 5, beta=1)
		if self.training:
			z = utils.reparametrize(mu, std, self.device)
		else:
			z = mu.clone().unsqueeze(0)
		decoded = self.decoder(z)
		return (mu, std), decoded[0]

class FMNIST_GMM_VAE(nn.Module):
	"""
	GMM variational auto encoder (not IB)
	"""
	def __init__(self, k, w, device):
		super(FMNIST_GMM_VAE, self).__init__()
		self.device = device
		self.description = 'GMM VAE'
		self.k = k
		self.w = w
		self.num_weights = w
		self.num_stochastic = 1
		self.part_len = k // w
		assert k % w == 0
		self.train_loss = []
		self.test_loss = []

		self.encoder = nn.Sequential(
			nn.Linear(784, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 2 * self.k * self.w))

		self.gmm = MixtureModle(w)

		self.decoder = nn.Sequential(
				nn.Linear(self.k, 1024),
				nn.ReLU(True),
				nn.Linear(1024, 28*28),
				nn.Sigmoid())

		# Xavier initialization
		for name, module in self._modules.items():
			if name == 'gmm':
				continue
			for layer in module:
				if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
							nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
							layer.bias.data.zero_()

	def forward(self, x):
		# squiwsh from shape (100,1,28,28) to (100,784)
		x = x.view(x.size(0),-1)
		z_params = self.encoder(x)

		mu = torch.stack(torch.split(z_params[:, :(self.k * self.w)], self.w, dim=-1), dim=2)
		# softplus transformation (soft relu) and a -5 bias is added as in the paper
		std = F.softplus(z_params[:, self.k * self.w:] - 5, beta=1)
		std = torch.stack(torch.split(std, self.w, dim=-1), dim=2)
		if self.training:
			z = utils.reparametrize(mu, std, self.device)
		else:
			z = mu.clone().unsqueeze(0)
		z, weights = self.gmm(z)
		decoded = self.decoder(z)
		return (mu, std), decoded[0], weights

class FMNIST_2STOCHASTIC_VAE(nn.Module):
	"""
	Just your regular variational auto encoder (not IB)
	"""
	def __init__(self, k, device):
		super(FMNIST_2STOCHASTIC_VAE, self).__init__()
		self.device = device
		self.description = '2 sequential stochastic layers VAE'
		self.k = k
		self.num_weights = 1
		self.num_stochastic = 2
		self.train_loss = []
		self.test_loss = []

		self.encoder = nn.Sequential(
			nn.Linear(784, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 2 * self.k))

		self.decoder = nn.Sequential(
				nn.Linear(self.k//2, 1024),
				nn.ReLU(True),
				nn.Linear(1024, 28*28),
				nn.Sigmoid())

		# Xavier initialization
		for _, module in self._modules.items():
			for layer in module:
				if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
							nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
							layer.bias.data.zero_()

	def forward(self, x):
		# squiwsh from shape (100,1,28,28) to (100,784)
		x = x.view(x.size(0),-1)
		z1_params = self.encoder(x)
		
		mu1 = z1_params[:, :self.k]
		std1 = F.softplus(z1_params[:, self.k:] - 5, beta=1)
		if self.training:
			z1 = utils.reparametrize(mu1, std1, self.device).squeeze(0)
		else:
			# z1 = mu1.clone().unsqueeze(0)
			z1 = mu1.clone()
		mu2 = z1[:, :self.k//2]
		std2 = F.softplus(z1[:, self.k//2:] - 5, beta=1)
		if self.training:
			z2 = utils.reparametrize(mu2, std2, self.device)
		else:
			z2 = mu2.clone().unsqueeze(0)
		decoded = self.decoder(z2)
		return (mu1, std1), (mu2, std2), decoded[0]