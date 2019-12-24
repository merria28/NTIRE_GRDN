import numpy as np
import cv2
import torch
from models.DenoisingModels import *
from utils.utils import *
from utils.transforms import *
import scipy.io as sio
import time
import tqdm

if __name__ == '__main__':

    print('********************Test code for NTIRE challenge******************')

    # path of input .mat file
    mat_dir = 'mats/BenchmarkNoisyBlocksRaw.mat'

    # Read .mat file
    mat_file = sio.loadmat(mat_dir)

    # get input numpy
    noisyblock = mat_file['BenchmarkNoisyBlocksRaw']
    
    print('input shape', noisyblock.shape)

    # path of saved pkl file of model
    modelpath = 'checkpoints/swjung_track1.pkl'
    expname = 'swjung_track1'

    # set gpu
    device = torch.device('cuda:0')

    # make network object
    model = ntire_rdb_gd_rir_ver2(input_channel=1, numofmodules=2, numforrg=4, numofrdb=16, numofconv=8, numoffilters=83).to(device)

    # make numpy of output with same shape of input
    resultNP = np.ones(noisyblock.shape)
    print('resultNP.shape', resultNP.shape)

    submitpath = f'results_folder/{expname}'
    make_dirs(submitpath)

    # load checkpoint of the model
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['state_dict'])

    transform = ToTensor()
    revtransform = ToImage()

    # pass inputs through model and get outputs
    with torch.no_grad():
        model.eval()
        starttime = time.time()     # check when model starts to process
        for imgidx in tqdm.tqdm(range(noisyblock.shape[0])):
            for patchidx in range(noisyblock.shape[1]):
                img = noisyblock[imgidx][patchidx]   # img shape (256, 256, 3)

                input = transform(img).float()
                input = input.view(1, -1, input.shape[1], input.shape[2]).to(device)

                output = model(input)       # pass input through model

                outimg = revtransform(output)   # transform output tensor to numpy

                # put output patch into result numpy
                resultNP[imgidx][patchidx] = outimg

    # check time after finishing task for all input patches
    endtime = time.time()
    elapsedTime = endtime - starttime   # calculate elapsed time
    print('ended', elapsedTime)
    num_of_pixels = noisyblock.shape[0] * noisyblock.shape[1] * noisyblock.shape[2] * noisyblock.shape[3]
    print('number of pixels', num_of_pixels)
    runtime_per_mega_pixels = (num_of_pixels / 1000000) / elapsedTime
    print('Runtime per mega pixel', runtime_per_mega_pixels)

    # save result numpy as .mat file
    sio.savemat(f'{submitpath}/{expname}', dict([('results', resultNP)]))
