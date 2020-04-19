import torch
from torch.utils.data import Dataset
import numpy as np

class BengaliAiDataset(Dataset):
    def __init__(self, imgs, labels):        
        super(BengaliAiDataset, self).__init__()                       
        self.imgs = imgs
        self.labels = labels              
    def __len__(self):
        return self.imgs.shape[0]            
    def __getitem__(self, index):         
        img = self.imgs[index,:,:].copy()                                                     
        img = img[:,:,np.newaxis]
        img = np.concatenate([img,img,img],axis=2)   
        img = (img/255).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
                        
        grapheme_root_label = torch.tensor(self.labels[index][0].copy()).long()
        vowel_diacritic_label = torch.tensor(self.labels[index][1].copy()).long()
        consonant_diacritic_label = torch.tensor(self.labels[index][2].copy()).long()
        
        return img,grapheme_root_label,vowel_diacritic_label,consonant_diacritic_label