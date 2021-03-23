import sys
import os
import time
import numpy as np
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader import get_eval_loader
from utils.plotting import generate_images, PDF_and_metrics
from networks import UNet

def adjust_LR(optimizer, params, iternum):
  """Piecewise constant rate decay"""
  if params.distributed and iternum<5000:
    lr = params.lr*(iternum/5000.) #warmup for distributed training
  
  elif iternum<params.LRsched[0]:
    lr = params.lr
  elif iternum>params.LRsched[1]:
    lr = params.lr/4.
  else:
    lr = params.lr/2.
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr



def eval(params, args, world_rank):
  loaders = [get_eval_loader(params, p, world_rank) for p in params.OODpaths]
  writers = [SummaryWriter(log_dir=os.path.join(expDir, name)) for name in params.OODnames]
  model = UNet.UNet(params).cuda()

  checkpoint = None
  if world_rank==0:
    logging.info("Loading checkpoint %s"%args.weights)
    checkpoint = torch.load(args.weights, map_location='cuda:{}'.format(args.local_rank))
    model.load_state_dict(checkpoint['model_state'])

  if world_rank==0:
    logging.info(model)
    logging.info("Starting Eval Loop...")

  device = torch.cuda.current_device()
  for i, name in enumerate(params.OODnames):
    loader = loaders[i]
    writer = writers[i]

    start = time.time()
    for iters in range(10):
      log_start = time.time()
      gens = []
      tars = []
      with torch.no_grad():
        for data in loader:
          inp, tar = map(lambda x: x.to(device), data)
          gen = model(inp)
          gens.append(gen.detach().cpu().numpy())
          tars.append(tar.detach().cpu().numpy())
      gens = np.concatenate(gens, axis=0)
      tars = np.concatenate(tars, axis=0)

      writer.add_scalar('Loss/test', np.mean(np.abs(gens - tars)), iters)

      fig, chi, KLscore = PDF_and_metrics(tars, gens, params)
      writer.add_figure('pixhist', fig, iters, close=True)
      writer.add_scalar('Metrics/chi', chi, iters)
      fig = generate_images(inp[-1].detach().cpu().numpy(), gens[-1], tars[-1], params.diff_precision)
      writer.add_figure('genimg', fig, iters, close=True)
    
    end = time.time()
    if world_rank==0:
        logging.info('Time taken for %s is %f sec'%(name, end-start))

  for w in writers:
    w.flush()
    w.close()



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--local_rank", default=0, type=int)
  parser.add_argument("--weights", default='./', type=str)
  parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  args = parser.parse_args()
  
  params = YParams(os.path.abspath(args.yaml_config), args.config)

  params.distributed = False
  if 'WORLD_SIZE' in os.environ:
    params.distributed = int(os.environ['WORLD_SIZE']) > 1

  world_rank = 0
  if params.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.gpu = args.local_rank
    world_rank = torch.distributed.get_rank() 

  torch.backends.cudnn.benchmark = True

  args.resuming = False

  # Set up directory
  baseDir = './OODtests/'
  expDir = os.path.join(baseDir, args.config)
  if  world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir)
  
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    params.log()

  params.experiment_dir = os.path.abspath(expDir)

  eval(params, args, world_rank)
  logging.info('DONE ---- rank %d'%world_rank)

