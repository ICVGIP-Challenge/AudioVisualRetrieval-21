import torch
import h5py
import os
import torch.nn.functional as F
import numpy as np

def read_h5(path):
	'''
	read h5 file. install h5py package
	'''
	hf = h5py.File(path, 'r')
	data = hf['data'][()]
	hf.close()
	
	return data

def parse_all_data(root, mode, classes, use_txt_feat=False, nexmp_per_class=None):
	'''
	parse both audio and video data for all class given root path
	along with the labels
	'''
	root_dir_audio = os.path.join(root, 'audio', mode)
	root_dir_video = os.path.join(root, 'video', mode)
	root_dir_text = os.path.join(root, 'text/word_embeddings-dict-33.npy')
	
	audioAll, videoAll, textAll, labelAll = [], [], [], []
	for i, target in enumerate(classes):
		print(target)
		# Load audio and video data
		data_audio = torch.from_numpy(read_h5(os.path.join(root_dir_audio, target+'.h5')))
		data_video = torch.from_numpy(read_h5(os.path.join(root_dir_video, target+'.h5')))
		
		# Permute data randomly
		perm = np.arange(data_audio.shape[0])
		np.random.shuffle(perm)
		data_audio, data_video = data_audio[perm], data_video[perm]
		
		if nexmp_per_class is not(None):
			if nexmp_per_class < data_audio.shape[0]:
				data_audio, data_video = data_audio[:nexmp_per_class], data_video[:nexmp_per_class]
		
		label = torch.tensor(classes.index(target)).expand(data_audio.shape[0])
		if use_txt_feat:
			text_embeddings = np.load(root_dir_text, allow_pickle=True).item()
			text_embed = torch.from_numpy(np.tile(text_embeddings[target], (data_audio.shape[0], 1)))
			textAll.append(text_embed)
		
		audioAll.append(data_audio)
		videoAll.append(data_video)
		labelAll.append(label)

	audio, video, label = torch.cat(audioAll, dim=0), torch.cat(videoAll, dim=0), torch.cat(labelAll, dim=0)
	
	if use_txt_feat:
		text = torch.cat(textAll, dim=0)
		return audio, video, text, label

	return audio, video, label

def calculate_precision_map(dist, labels, start_idx, end_idx, k_val=[1, 5, 10], reduced=False):
	'''
	Calculate the precision given the distance matrix

	input: dist, lables
	dist: torch.tensor of size NxN, where N is the no. of examples
	labels: torch.tensor of size Nx1 (label for each of the query)
	k_val: specific values for calculating p@k
	reduced: boolean value to whether average over all the values

	output: precision, mean_avg_prcsn
	precision: precision vlaues at specified index
	mean_avg_prcsn: map values
	'''

	if dist.shape[1] != labels.shape[0]:
		raise ValueError('Dimensions are not correct....')
	index = torch.argsort(dist, dim=1)
	
	matches = (labels[index] == labels[start_idx:end_idx, None])
	num_rel = matches.sum(1).float()

	tmp_prcsn = matches.cumsum(1).float()
	norm_div = torch.Tensor([i+1 for i in range(tmp_prcsn.shape[1])]).cuda()
	tmp_prcsn = torch.div(tmp_prcsn, norm_div)
	precision = {}

	if reduced:
		for k in k_val:
			precision[k] = torch.mean(tmp_prcsn[:, k-1])
			
	if not(reduced):
		for k in k_val:
			precision[k] = tmp_prcsn[:, k-1]

	map_tmp = tmp_prcsn * matches.float()
	AP = map_tmp.sum(1) / num_rel
	if reduced:
		mean_avg_prcsn = torch.mean(AP)
		
	if not(reduced):
		mean_avg_prcsn = AP

	return precision, mean_avg_prcsn

def calculate_distance(query, gallery, norm='euclidean'):
	'''
	function to calculate pairwisw distance between two matrices.

	input: query, gallery matirx
	query: matix of size NxD
	gallery: matrix of size MxD

	output: dist matrix
	dist: matrix of size NxM
	'''
	if norm == 'cosine':
		query = F.normalize(query, p=2, dim=1)
		gallery = F.normalize(gallery, p=2, dim=1)

	n = query.size(0)
	m = gallery.size(0)
	d = query.size(1)

	query = query.unsqueeze(1).expand(n, m, d)
	gallery = gallery.unsqueeze(0).expand(n, m, d)
	
	dist = torch.pow(query - gallery, 2).sum(2) 

	return dist

def calculate_classwise_metric(metric, label):
	'''
	calculate the classwise average for all the retrieval results
	input: metric, label
	metric: individual metric for every examples
	label: class of every example

	output: class average of map
	'''
	metric_classwise = torch.zeros(int(torch.max(label)+1))
	count_classwise = torch.zeros(int(torch.max(label)+1))
	for q in range(label.shape[0]):
		metric_classwise[int(label[q])] += metric[q]
		count_classwise[int(label[q])] += 1
	
	indx = torch.nonzero(count_classwise).squeeze()
	metric_norm = metric_classwise[indx]/count_classwise[indx]

	return torch.mean(metric_norm)

def calculate_both_map(audio_val, video_val, label_val, gpu=True, batchSizeEval = 4, out_txt=False):
	'''
	find the average of audio2video and video2audio retrieval
	
	input: audio_val, video_val, label_bal
	audio_val: matrix of audio features
	video_val: matrix of video features
	label_val: labels of every example
	
	output: 
	audio2video and video2audio retrieval
	'''
	Ndata = audio_val.shape[0]

	if gpu:
		audio_val = audio_val.cuda()
		video_val = video_val.cuda()
		label_val = label_val.cuda()

	map_aud2vid_all = []
	map_vid2aud_all = []
	if out_txt:
		path_aud2vid = './aud2vid.txt'
		path_vid2aud = './vid2aud.txt'
		file_aud2vid = open(path_aud2vid, 'w')
		file_vid2aud = open(path_vid2aud, 'w')
	for i in range(0, Ndata, batchSizeEval):
		startidx = i
		endidx = startidx+batchSizeEval
		if endidx > Ndata:
			endidx = Ndata

		## audio to video retrieval
		dist_aud2vid = calculate_distance(audio_val[startidx:endidx], video_val)
		if out_txt:
			write_txt(file_aud2vid, dist_aud2vid, label_val, startidx, endidx, gpu)
		_, map_aud2vid = calculate_precision_map(dist_aud2vid, label_val, startidx, endidx)
		if gpu:
			map_aud2vid_all += map_aud2vid.cpu()
		else:
			map_aud2vid_all += map_aud2vid

		## video to audio retrieval
		dist_vid2aud = calculate_distance(video_val[startidx:endidx], audio_val)
		if out_txt:
			write_txt(file_vid2aud, dist_vid2aud, label_val, startidx, endidx, gpu)
		_, map_vid2aud = calculate_precision_map(dist_vid2aud, label_val, startidx, endidx)
		if gpu:
			map_vid2aud_all += map_vid2aud.cpu()
		else:
			map_vid2aud_all += map_vid2aud

	AvgMAP = {}
	
	AvgMAP['aud2vid'] = calculate_classwise_metric(map_aud2vid_all, label_val)
	AvgMAP['vid2aud'] = calculate_classwise_metric(map_vid2aud_all, label_val)
	AvgMAP['avg'] = 0.5*(AvgMAP['aud2vid'] + AvgMAP['vid2aud'])
	
	if out_txt:
		file_aud2vid.close()
		file_vid2aud.close()

	return AvgMAP

def write_txt(file, dist, label, start_idx, end_idx, gpu):
	if dist.shape[1] != label.shape[0]:
		raise ValueError('Dimensions are not correct....')
	index = torch.argsort(dist, dim=1)
	labels_to_write = torch.cat((label[start_idx:end_idx].unsqueeze(1), label[index]), 1)

	if gpu:
		labels_to_write = labels_to_write.cpu()

	for i in range(start_idx, end_idx):
		str_to_write = ','.join(str(i) for i in labels_to_write[i-start_idx].tolist())+'\n'
		file.write(str_to_write)
	

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'