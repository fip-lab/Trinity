import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['AVEDataset', 'AVEDataLoader']




class AVEDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args

        with open(args.dataPath, 'rb') as f:
            data = pickle.load(f)

        data = data[mode]
        self.ids = data['id']
        self.labels = np.array(data['label'], dtype=np.int64)
        self.strong_labels = np.array(data['strong_label'], dtype=np.float32)

        # Load clues
        self.visual_clue_bert = np.array(data['Visual_cues_bert'], dtype=np.float32)
        self.audio_clue_bert = np.array(data['Audio_cues_bert'], dtype=np.float32)
        self.text_time_bert = np.array(data['Time_text_bert'], dtype=np.float32)

        # Feature modalities
        self.vision = np.array(data[args.vision_feats], dtype=np.float32)
        self.audio = np.array(data[args.audio_feats], dtype=np.float32)

        logger.info(f"[{mode}] before truncation vision={self.vision.shape}, audio={self.audio.shape}")

        # 清理脏数据
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0

        if self.args.need_truncated:
            self.__truncated()

        if self.args.need_normalize:
            self.__normalize()

        logger.info(f"[{mode}] after process vision={self.vision.shape}, audio={self.audio.shape}, v_clue={self.visual_clue_bert.shape}, a_clue={self.audio_clue_bert.shape}, t_clue={self.text_time_bert.shape}")



    def __normalize(self):
        print("Normalizing data...")
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))



    def __truncated(self):
        print("Truncating data...")
        # NOTE: Here for dataset we manually cut the input into specific length.
        # NOTE 这里我们手动将输入切割成特定长度。
        def TruncatedText(text_features, length):
            # text_features 的形状为 (batch_size, 3, seq_len)
            if text_features.shape[2] > length:
                return text_features[:, :, :length]
            # 如果长度不足，进行零填充
            padding = np.zeros((text_features.shape[0], text_features.shape[1], length - text_features.shape[2]))
            return np.concatenate((text_features, padding), axis=2)
        
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+length])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+length])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens

        self.vision = Truncated(self.vision, video_length)
        self.audio = Truncated(self.audio, audio_length)

        # clue
        self.audio_clue_bert = TruncatedText(self.audio_clue_bert, text_length)
        self.visual_clue_bert = TruncatedText(self.visual_clue_bert, text_length)
        self.text_time_bert = TruncatedText(self.text_time_bert, text_length)



    def __len__(self):
        return len(self.ids)

    def get_seq_len(self):
        return (self.vision.shape[1], self.audio.shape[1])

    def get_feature_dim(self):
        return self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        return {
            'id': self.ids[index],
            'vision': torch.tensor(self.vision[index], dtype=torch.float),
            'audio': torch.tensor(self.audio[index], dtype=torch.float),
            'index': index,
            'label': torch.tensor(self.labels[index], dtype=torch.long),
            'strong_label': torch.tensor(self.strong_labels[index], dtype=torch.float),
            'visual_clue_bert': torch.tensor(self.visual_clue_bert[index], dtype=torch.float),
            'audio_clue_bert': torch.tensor(self.audio_clue_bert[index], dtype=torch.float),
            'time_text_bert': torch.tensor(self.text_time_bert[index], dtype=torch.float)
        }



def AVEDataLoader(args, mode='train'):
    if mode == 'train':
        datasets = {
            'train': AVEDataset(args, 'train'),
            'valid': AVEDataset(args, 'valid'),
            'test': AVEDataset(args, 'test')
        }
    else:
        datasets = {
            'test': AVEDataset(args, 'test')
        }
    
    args.ave_seq_lens = datasets['test'].get_seq_len()

    return {
        ds: DataLoader(
            datasets[ds],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        ) for ds in datasets
    }