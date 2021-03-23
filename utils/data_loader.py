import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor
import h5py


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))


def get_data_loaders_distributed(params, world_rank):

    train_dataset = SliceDataset(params, isTrain=True)
    val_dataset = SliceDataset(params, isTrain=False)

    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=params.batch_size,
                              num_workers=params.num_data_workers,
                              worker_init_fn=worker_init,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset,
                            shuffle=True,
                            batch_size=params.batch_size,
                            num_workers=params.num_data_workers,
                            worker_init_fn=worker_init,
                            pin_memory=torch.cuda.is_available())
    return train_loader, val_loader



def get_eval_loader(params, path, world_rank):

    dataset = SliceDataset(params, isTrain=False, OODpath=path)
    loader = DataLoader(dataset,
                        shuffle=True,
                        batch_size=params.batch_size,
                        num_workers=params.num_data_workers,
                        worker_init_fn=worker_init,
                        pin_memory=torch.cuda.is_available())
    return loader




class SliceDataset(Dataset):

    def __init__(self, params, isTrain=True, OODpath=None):
        self.isOOD = OODpath is not None
        self.fname = params.data_path if not self.isOOD else OODpath
        self.sz = params.data_size
        self.time = params.time_idx
        self.isTrain = isTrain
        self.b, self.s = params.bias_scale
        if isTrain:
          self.Nsamples = params.Ntrain
        elif self.isOOD:
          self.Nsamples = h5py.File(self.fname+'_Dc1.h5', 'r')['phi'].shape[0]
        else:
          self.Nsamples = params.Nval
        self.Dc1, self.Dc2 = [None, None]

    def __len__(self):
        return self.Nsamples

    def _open_files(self):
        self.Dc1 = h5py.File(self.fname+'_Dc1.h5', 'r')
        self.Dc2 = h5py.File(self.fname+'_Dc2.h5', 'r')

    def __getitem__(self, idx):
        if not self.Dc1:
            self._open_files()

        offs = 0 if self.isTrain else -self.Nsamples
        inp = self.Dc1['phi'][offs+idx,:,:,self.time] + self.b
        tar = self.Dc2['phi'][offs+idx,:,:,self.time] + self.b
        inp = self.s*np.moveaxis(inp, -1, 0).astype(np.float32)
        tar = self.s*np.moveaxis(tar, -1, 0).astype(np.float32)
        return torch.as_tensor(inp), torch.as_tensor(tar)


class RandomDataset(Dataset):

    def __init__(self, params, isTrain=True):
        self.fname = params.data_path
        self.sz = params.data_size
        self.time = params.time_idx
        self.Nsamples = params.Ntrain if isTrain else params.Nval
        self.x, self.y = np.meshgrid(np.linspace(-1., 1., num=self.sz), np.linspace(-1., 1., num=self.sz))
        self.dx = 2./self.sz

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):

        x0, y0 = np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)
        r2 = (self.x - x0)**2 + (self.y - y0)**2
        amp = np.random.uniform(0.1, 0.9)
        wd1 = np.random.uniform(2.*self.dx, 10*self.dx)
        wd2 = 0.8*wd1
        
        Dc1 = amp*np.exp(-(r2/(wd1**2)))
        Dc2 = amp*np.exp(-(r2/(wd2**2)))

        inp = np.expand_dims(Dc1, axis=0).astype(np.float32)
        tar = np.expand_dims(Dc2, axis=0).astype(np.float32)
        return torch.as_tensor(inp), torch.as_tensor(tar)


