from utils.util import calculate_both_map, parse_all_data
from utils.cca import CCA 
import torch
import torch.nn.functional as F
import numpy as np
import argparse, os
import h5py

parser = argparse.ArgumentParser(description='Cross Modal Retrieval Training')
parser.add_argument('-path', metavar='DIR', default = './data',
					help='path to dataset')
parser.add_argument('-mode', default = 'raw', help='type of baselines: cca/raw')
parser.add_argument('-n', '--normalize', dest='normalize', action='store_true',
					help='normalize the input data')
parser.add_argument('-c', '--cosine', dest='cosine', action='store_true',
					help='calcualte cosine distance')

def main():
	args = parser.parse_args()
	args.gpu=torch.cuda.is_available()
	## parameters
	root_path = os.path.join(args.path, 'features')
	class_labels_path = os.path.join(args.path, 'class-split/seen_class.txt')
	file = open(class_labels_path, 'r')
	class_labels = [cls_name.strip(",\n'") for cls_name in file.readlines()]
	

	## load train data
	audio_trn, video_trn, text_trn, label_trn = parse_all_data(root_path, 'trn', class_labels, use_txt_feat=True)
	if args.normalize:
		audio_trn = F.normalize(audio_trn, p=2, dim=1)
		video_trn = F.normalize(video_trn, p=2, dim=1)
		text_trn = F.normalize(text_trn, p=2, dim=1)

	## load val data
	path = os.path.join(root_path, 'val.h5')
	hf = h5py.File(path, 'r')
	audio_val, video_val, label_val = torch.from_numpy(hf['audio'][()]), torch.from_numpy(hf['video'][()]), torch.from_numpy(hf['label'][()])
	hf.close()

	if args.normalize:
		audio_val = F.normalize(audio_val, p=2, dim=1)
		video_val = F.normalize(video_val, p=2, dim=1)

	if args.mode == 'cca':
		cca = CCA()
		cca.fit(audio_trn.numpy(), video_trn.numpy())
		audio_val, video_val = cca.transform(audio_val.numpy(), video_val.numpy())
		audio_val, video_val = torch.from_numpy(audio_val), torch.from_numpy(video_val)

	if args.cosine:
		audio_val = F.normalize(audio_val, p=2, dim=1)
		video_val = F.normalize(video_val, p=2, dim=1)

	if args.gpu:
		audio_val, video_val = audio_val.cuda(), video_val.cuda()

	## evaluate on val data
	AvgMap = calculate_both_map(audio_val, video_val, label_val, gpu=args.gpu, out_txt=True)
	print('aud2vid:{}\tvid2aud:{}\tAvg:{}'.format(AvgMap['aud2vid'], AvgMap['vid2aud'], AvgMap['avg']))

if __name__ == '__main__':
	main()
