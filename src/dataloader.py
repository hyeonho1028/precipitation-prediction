from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
# class RainDataset(Dataset):
#     def __init__(self, config, imgs, label, indicies , mode):
#         self.config = config
#         self.imgs = imgs
#         self.label = label
#         self.indicies = np.asarray( indicies )
#         self.mode = mode
        
#     def __len__(self):
#         return len( self.indicies )
    
#     def __getitem__(self, idx):        
#         idx = self.indicies[idx]
        
#         # Load Image
#         imgs = self.imgs[idx, :, :, :]
#         imgs = imgs.astype(np.float32)
# #         img = cv2.resize( img, ( self.config.resize_width, self.config.resize_height ) )
    
# #         Scaling
#         imgs = np.log1p(imgs)    
        
#         # Label
#         if self.mode == "train":            
#             label = self.label[idx, :, :, :]
#             label = label.astype(np.float32)
#             return imgs, label
#         else:
#             return img

class RainDataset(Dataset):
    def __init__(self, config, img_path=[], label_path=[], mode='train'):
        self.config = config
        self.img_path = img_path
        self.label_path = label_path
        if mode!='test':
            # self.image = np.expand_dims(np.concatenate([np.expand_dims(np.load(sp, allow_pickle=True)[:, :, 0], 0) for sp in self.path]), -1)
            # self.label = np.expand_dims(np.concatenate([np.expand_dims(np.load(sp, allow_pickle=True)[:, :, 1], 0) for sp in self.path]), -1)
            self.image = np.concatenate([np.expand_dims(np.load(sp, allow_pickle=True)[:, :, :4], 0) for sp in self.img_path])
            self.label = np.concatenate([np.expand_dims(np.load(sp, allow_pickle=True)[:, :, 4], 0) for sp in self.label_path])
            self.label = np.expand_dims(self.label, -1)
        else:
            self.image = np.concatenate([np.expand_dims(np.load(sp, allow_pickle=True)[:, :, :], 0) for sp in self.img_path])
        self.mode = mode

        if mode!='test':
            # 106 * 106
            crop_size=7
            # self.image = self.image[:, crop_size:-crop_size, crop_size:-crop_size, :]
            # self.label = self.label[:, crop_size:-crop_size, crop_size:-crop_size, :]
            for idx in [[67, 28], [64, 46], [69, 57], [66, 62], [64, 68], [71, 70], [87, 56]]:
                self.label[:, idx[0], idx[1]] = self.label[:, idx[0], idx[1]]*2
            print(self.image.shape, self.label.shape)
            self.image = np.transpose(self.image, (0,3,1,2))
            self.label = np.transpose(self.label, (0,3,1,2))

            self.image = np.where(self.image<0, 0, self.image)
            self.label = np.where(self.label<0, 0, self.label)

            self.image = np.log1p(self.image)
        else:
            # self.image = self.image[:, crop_size:-crop_size, crop_size:-crop_size, :]
            # for idx in [[67, 28], [64, 46], [69, 57], [66, 62], [64, 68], [71, 70], [87, 56]]:
            #     self.image[:, idx[0], idx[1]] = self.image[:, idx[0], idx[1]]*2
            print(self.image.shape)
            self.image = np.transpose(self.image, (0,3,1,2))
            self.image = np.where(self.image<0, 0, self.image)
            self.image = np.log1p(self.image)
        
    def __len__(self):
        return len( self.img_path )
    
    def __getitem__(self, idx):
        # Load Image
        imgs = self.image[idx]
        imgs = imgs.astype(np.float32)
    
#         img = cv2.resize( img, ( self.config.resize_width, self.config.resize_height ) )
        
        # Label
        if self.mode != "test":
            label = self.label[idx]
            label = label.astype(np.float32)
            return imgs, label
        else:
            return imgs