import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.util import parse_all_data
import pandas
import h5py

class EvalDatasetAV(Dataset):
	def __init__(self, root, mode ='val'):
		path = os.path.join(root, 'features', mode+'.h5')
		hf = h5py.File(path, 'r')
		self.audio_data_all, self.video_data_all, self.label_all = hf['audio'][()], hf['video'][()], hf['label'][()]
		hf.close()

	def __getitem__(self, index):	
		audio = self.audio_data_all[index]
		video = self.video_data_all[index]
		label = self.label_all[index]
		

		data = {'audio':audio, 'video':video, 'label':label}

		return data
		
	def __len__(self):
		return self.audio_data_all.shape[0]


class tripletDatasetAV(Dataset):
	def __init__(self, root, labels_txt, mode ='trn'):
		
		self.audio_data_all, self.video_data_all, self.label_all = parse_all_data(os.path.join(root, 'features'), mode, labels_txt)
				
		self.text_embeddings = np.load(os.path.join(root, 'features/text/word_embeddings-dict-33.npy'), allow_pickle=True).item()
		self.class_names = labels_txt

		pos_neg_cls_path = os.path.join(root, 'triplet_split/triplets_zeroshot_True.csv')
		self.cls_slctd = pandas.read_csv(pos_neg_cls_path, header=None)
		
	def __getitem__(self, index):
		
		pos_cls, neg_cls = self.cls_slctd.iloc[index].tolist()
		
		pos_text = torch.from_numpy(self.text_embeddings[pos_cls]) 
		neg_text = torch.from_numpy(self.text_embeddings[neg_cls])
		
		pos_index = np.asarray(np.where(self.label_all == self.class_names.index(pos_cls)))
		pos_sample_index = np.random.choice(pos_index[0])
		
		neg_index = np.asarray(np.where(self.label_all == self.class_names.index(neg_cls)))
		neg_sample_index = np.random.choice(neg_index[0])
		
		
		pos_audio, neg_audio = self.audio_data_all[pos_sample_index], self.audio_data_all[neg_sample_index]
		pos_video, neg_video = self.video_data_all[pos_sample_index], self.video_data_all[neg_sample_index]

		data = {}
		data['pos_audio'], data['neg_audio'] = pos_audio, neg_audio
		data['pos_video'], data['neg_video'] = pos_video, neg_video
		data['pos_text'], data['neg_text'] = pos_text, neg_text
		data['pos_class'], data['neg_class'] = pos_cls, neg_cls
		
		return data

	def __len__(self):
		return self.cls_slctd.shape[0]