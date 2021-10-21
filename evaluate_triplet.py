import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import os
from dataloader.data_loader import tripletDatasetAV, EvalDatasetAV
from torch.utils.data import DataLoader 
from utils.util import parse_all_data, calculate_both_map, AverageMeter, ProgressMeter
from models.networks import build_network, CombinedNetworkAV
import random
import numpy as np

parser = argparse.ArgumentParser(description='Cross Modal Retrieval Training')
parser.add_argument('-path', metavar='DIR', default = './data', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('-n', '--normalize', dest='normalize', action='store_true', help='normalize the input data')
parser.add_argument('-c', '--cosine', dest='cosine', action='store_true', help='calcualte cosine distance')
parser.add_argument('-chk_dir', '--checkpoints_dir', metavar='DIR', default = './checkpoints', help='path to dataset')


def main():
	args = parser.parse_args()
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	args.gpu=torch.cuda.is_available()
	
	## parameters
	file = open(os.path.join(args.path, 'class-split/seen_class.txt'), 'r')
	class_labels = [cls_name.strip(",\n'") for cls_name in file.readlines()]

	arch_audio = [1024, 512, 256]
	arch_video = [1024, 512, 256]
	arch_text = [300, 256]

	## dataloader
	dataset_val = EvalDatasetAV(args.path, mode ='test_with_shuffle_label')
	dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	
	##model
	#model is defined here
	netAudio = build_network(arch_audio, weights=os.path.join(args.checkpoints_dir, 'audio.pth'))
	netVideo = build_network(arch_video, weights=os.path.join(args.checkpoints_dir, 'video.pth'))
	model = CombinedNetworkAV(netAudio, netVideo)

	if torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
	model.to(args.device)
	
	print('======Evaluating======')

	# switch to evaluate mode
	model.eval()
	audioAll, videoAll, labelAll = [], [], []
	with torch.no_grad():
		for i, data in enumerate(dataloader_val):
			audio = data['audio'].to(args.device) 
			video = data['video'].to(args.device)
			label = data['label'].to(args.device)

			if args.normalize:
				audio = F.normalize(audio, p=2, dim=1)
				video = F.normalize(video, p=2, dim=1) 
				
			# Perfrom forward pass and compute the output
			out = model(audio, video)
			# import pdb; pdb.set_trace()
			
			if args.cosine:
				out['audio'], out['video'] = F.normalize(out['audio'], p=2, dim=1), F.normalize(out['video'], p=2, dim=1)
 
			audioAll.append(out['audio'])
			videoAll.append(out['video'])
			labelAll.append(label)

		audioAll, videoAll, labelAll = torch.cat(audioAll, dim=0), torch.cat(videoAll, dim=0), torch.cat(labelAll, dim=0)	
		# measure retrieval map
		AvgMap = calculate_both_map(audioAll, videoAll, labelAll, gpu=args.gpu, out_txt=True)
		
		print('aud2vid:{}\tvid2aud:{}\tAvg:{}'.format(AvgMap['aud2vid'], AvgMap['vid2aud'], AvgMap['avg']))

	
if __name__ == '__main__':
	main()