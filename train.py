from __future__ import absolute_import
from  __future__ import division
from __future__ import print_function

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import utils
from dataset import MNISTDataset
from fconv import CNNEncoder, CNNDecoder


_EPOCH = 50
_GPU_ID = 0
_BATCH_SIZE = 32
_IMAGE_SIZE = 64
_ENC_LEN = 10
data_path = 'data/mnist_test_seq.npy'
dataset = MNISTDataset(data_path)
loader = data.DataLoader(dataset, _BATCH_SIZE, shuffle=True)

# # show 1 batch of example
# vids = next(iter(loader))
# print(vids.size())
# for i in range(vids.size(0)):
#     utils.show_video(vids[i], 3)

# manually set the random seed for reproducibility.
torch.manual_seed(1)
str_time = time.strftime('%Y_%b_%d_%a_%H_%M', time.gmtime())
print(str_time)
save_path = os.path.join('save/', str_time)
os.makedirs(save_path)

encoder = CNNEncoder(4096, hid_size=1024, num_layers=3, enc_len=_ENC_LEN)
decoder = CNNDecoder(1024, output_size=4096, hid_size=1024, num_layers=3)
encoder = encoder.cuda(_GPU_ID)
decoder = decoder.cuda(_GPU_ID)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-5)


loss_his = []
for e in range(_EPOCH):
    start_time = time.time()
    for i, sample in enumerate(loader):
        optimizer.zero_grad()
        sample = Variable(sample).cuda(_GPU_ID)
        enc_out = encoder(sample.float())
        dec_out = decoder(enc_out)
        recovered = dec_out.view(_BATCH_SIZE, _ENC_LEN, _IMAGE_SIZE, _IMAGE_SIZE)
        loss = loss_fn(recovered, sample.long())
        loss_his.append(float(loss[0].data.cpu().numpy()))

        if i % 100 == 0:
            training_time = (time.time() - start_time) / 60.0
            loss_cur = sum(loss[-100:])/100 if i != 0 else loss[-1]
            log_entry = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | minutes {:5.2f} | loss {:5.6f} |'.\
                format(e, i, len(loader), optimizer.param_groups[0]['lr'],training_time, 10.0 * loss_cur)
            print(log_entry)
            with open(os.path.join(save_path, 'train.log'), 'a') as fobj:
                fobj.write(log_entry + '\n')
    torch.save(encoder.cpu(), os.path.join(save_path, 'encoder.pth'))
    torch.save(decoder.cpu(), os.path.join(save_path, 'decoder.pth'))
    encoder = encoder.cuda(_GPU_ID)
    decoder = decoder.cuda(_GPU_ID)
