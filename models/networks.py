import torch.nn as nn
import torch

class LinearNet(nn.Module):
	def __init__(self, NetStruct):
		super(LinearNet, self).__init__()
		self.layers = nn.ModuleList()
		self.bn_layers = nn.ModuleList()
		for i in range(len(NetStruct) - 1):
			self.layers.append(nn.Linear(NetStruct[i],NetStruct[i+1]))
		for i in range(len(NetStruct) - 1):
			self.bn_layers.append(nn.BatchNorm1d(NetStruct[i+1]))
		self.relu = nn.ReLU()
	
	def forward(self, x):
		if len(self.layers) > 1:
			for i in range(len(self.layers)-1):
				x = self.layers[i](x)
				x = self.relu(x)
				x = self.bn_layers[i](x)
			y = self.layers[i+1](x)
		
		if len(self.layers) == 1:
			y = self.layers[0](x)

		return y   

def build_network(NetArch, weights=''):
	net = LinearNet(NetArch)

	if len(weights) > 0:
		print('Loading weights...')
		net.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
	return net

class CombinedNetwork(nn.Module):
	def __init__(self, audio_net, video_net, text_net):
		super(CombinedNetwork, self).__init__()
		self.audio_net = audio_net
		self.video_net = video_net
		self.text_net = text_net

	def forward(self, input_audio, input_video, input_text):
		audio_out = self.audio_net(input_audio)
		video_out = self.video_net(input_video)
		text_out = self.text_net(input_text)

		out = {}
		out['audio'], out['video'], out['text'] = audio_out, video_out, text_out

		return out


class CombinedNetworkAV(nn.Module):
	def __init__(self, audio_net, video_net):
		super(CombinedNetworkAV, self).__init__()
		self.audio_net = audio_net
		self.video_net = video_net
		
	def forward(self, input_audio, input_video):
		audio_out = self.audio_net(input_audio)
		video_out = self.video_net(input_video)
		
		out = {}
		out['audio'], out['video'] = audio_out, video_out

		return out