import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import os
from dataloader.data_loader import tripletDatasetAV, DatasetAV
from torch.utils.data import DataLoader 
from utils.util import parse_all_data, calculate_both_map, AverageMeter, ProgressMeter
from models.networks import build_network, CombinedNetwork


parser = argparse.ArgumentParser(description='Cross Modal Retrieval Training')
parser.add_argument('-path', metavar='DIR', default = './data',
					help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int, 
					help='print frequency (default: 100)')
parser.add_argument('-n', '--normalize', dest='normalize', action='store_true',
					help='normalize the input data')
parser.add_argument('-c', '--cosine', dest='cosine', action='store_true',
					help='calcualte cosine distance')
parser.add_argument('-val_on', '--validation_on', action='store_true',
					help='whether to evaluate model')
parser.add_argument('-val_freq', '--validation_freq', default=5, type=int, 
					help='perform validation (default: 5)')
parser.add_argument('-chk_dir', '--checkpoints_dir', metavar='DIR', 
					default = './checkpoints', help='path to dataset')
parser.add_argument('-lt', '--lambda-t', dest='lambda_t', default=1.0, type=float, 
					help='weights for triplet loss')
parser.add_argument('-lc', '--lambda-c', dest='lambda_c', default=1.0, type=float, 
					help='weights for cross modal loss')

def main():
	args = parser.parse_args()
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	## parameters
	file = open(os.path.join(args.path, 'class-split/seen_class.txt'), 'r')
	class_labels = [cls_name.strip(",\n'") for cls_name in file.readlines()]

	arch_audio = [1024, 512, 256]
	arch_video = [1024, 512, 256]
	arch_text = [300, 256]

	## dataloader
	# for loading the dataset 
	dataset_trn = tripletDatasetAV(args.path, class_labels, mode ='trn')
	dataset_val = DatasetAV(args.path, class_labels, mode ='val')

	dataloader_trn = DataLoader(dataset_trn, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
	dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	
	##model
	#model is defined here
	netAudio = build_network(arch_audio)
	netVideo = build_network(arch_video)
	netText = build_network(arch_text)
	model = CombinedNetwork(netAudio, netVideo, netText)

	if torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
	model.to(args.device)
	
	criterion = {}
	criterion['mse'] = nn.MSELoss(reduction='mean').to(args.device)
	criterion['triplet'] = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean').to(args.device)


	# optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	best_map = 0.0
	## training code 
	for epoch in range(args.epochs):
		adjust_learning_rate(optimizer, epoch, args)

		# train for one epoch
		train(dataloader_trn, model, criterion, optimizer, epoch, args)
		if (args.validation_on and (epoch+1) % args.validation_freq == 0): 
			avg_map = validate(dataloader_val, model, args)
			if avg_map > best_map:
				best_map = avg_map
				if not(os.path.exists(args.checkpoints_dir)):
					os.makedirs(args.checkpoints_dir)
				torch.save(netAudio.state_dict(), os.path.join(args.checkpoints_dir, 'audio.pth'))
				torch.save(netVideo.state_dict(), os.path.join(args.checkpoints_dir, 'video.pth'))
				torch.save(netText.state_dict(), os.path.join(args.checkpoints_dir, 'text.pth'))

		

def train(train_loader, model, criterion, optimizer, epoch, args):
	triplet_losses = AverageMeter('Triplet Loss', ':.4e')
	cross_modal_losses = AverageMeter('CM Loss', ':.4e')
	losses = AverageMeter('Loss', ':.4e')
	
	progress = ProgressMeter(
		len(train_loader), [triplet_losses, cross_modal_losses, losses],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	for i, data in enumerate(train_loader):
		# Transfer data to appropriate device
		pos_audio, neg_audio = data['pos_audio'].to(args.device), data['neg_audio'].to(args.device)
		pos_video, neg_video = data['pos_video'].to(args.device), data['neg_video'].to(args.device)
		pos_text, neg_text = data['pos_text'].to(args.device), data['neg_text'].to(args.device)

		if args.normalize:
			pos_audio, neg_audio = F.normalize(pos_audio, p=2, dim=1), F.normalize(neg_audio, p=2, dim=1)
			pos_video, neg_video = F.normalize(pos_video, p=2, dim=1), F.normalize(neg_video, p=2, dim=1)
			pos_text, neg_text = F.normalize(pos_text, p=2, dim=1), F.normalize(neg_text, p=2, dim=1)

		# Perfrom forward pass and compute the output
		out_pos = model(pos_audio, pos_video, pos_text)
		out_neg = model(neg_audio, neg_video, neg_text)

		if args.cosine:
			out_pos['audio'], out_neg['audio'] = F.normalize(out_pos['audio'], p=2, dim=1), F.normalize(out_neg['audio'], p=2, dim=1)
			out_pos['video'], out_neg['video'] = F.normalize(out_pos['video'], p=2, dim=1), F.normalize(out_neg['video'], p=2, dim=1)
			out_pos['text'], out_neg['text'] = F.normalize(out_pos['text'], p=2, dim=1), F.normalize(out_neg['text'], p=2, dim=1)

		# compute loss
		loss_audio_triplet = criterion['triplet'](out_pos['text'], out_pos['audio'], out_neg['audio'])
		loss_video_triplet = criterion['triplet'](out_pos['text'], out_pos['video'], out_neg['video'])
		loss_triplet = loss_audio_triplet + loss_video_triplet

		loss_crossmodal = criterion['mse'](pos_audio, pos_video) + criterion['mse'](neg_audio, neg_video)

		loss = args.lambda_t*loss_triplet + args.lambda_c*loss_crossmodal

		# measure accuracy and record loss
		losses.update(loss.item(), pos_audio.size(0))
		triplet_losses.update(loss_triplet.item(), pos_audio.size(0))
		cross_modal_losses.update(loss_crossmodal.item(), pos_audio.size(0))
		
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % args.print_freq == 0:
			progress.display(i)


def validate(val_loader, model, args):
	print('======Evaluating======')

	# switch to evaluate mode
	model.eval()
	audioAll, videoAll, labelAll = [], [], []
	with torch.no_grad():
		for i, data in enumerate(val_loader):
			audio = data['audio'].to(args.device) 
			video = data['video'].to(args.device)
			text = data['text'].to(args.device)
			label = data['label'].to(args.device)

			if args.normalize:
				audio = F.normalize(audio, p=2, dim=1)
				video = F.normalize(video, p=2, dim=1) 
				text = F.normalize(text, p=2, dim=1)
				
			# Perfrom forward pass and compute the output
			out = model(audio, video, text)
			# import pdb; pdb.set_trace()
			
			if args.cosine:
				out['audio'], out['video'] = F.normalize(out['audio'], p=2, dim=1), F.normalize(out['video'], p=2, dim=1)
 
			audioAll.append(out['audio'])
			videoAll.append(out['video'])
			labelAll.append(label)

		audioAll, videoAll, labelAll = torch.cat(audioAll, dim=0), torch.cat(videoAll, dim=0), torch.cat(labelAll, dim=0)	
		# measure retrieval map
		AvgMap = calculate_both_map(audioAll, videoAll, labelAll)
		
		print('aud2vid:{}\tvid2aud:{}\tAvg:{}'.format(AvgMap['aud2vid'], AvgMap['vid2aud'], AvgMap['avg']))

	return AvgMap['avg']

def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

if __name__ == '__main__':
	main()