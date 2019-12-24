import numpy as np
import matplotlib.pyplot as plt
import cv2
from models.DenoisingModels import *
import tqdm
from utils.utils import *
from utils.transforms import *
import scipy.io as sio
import time

if __name__ == '__main__':

    print('********************Test code for NTIRE challenge******************')

    # path of input .mat file
    mat_dir = 'mats/BenchmarkNoisyBlocksSrgb.mat'
    # Read .mat file
    mat_file = sio.loadmat(mat_dir)

    # get input numpy
    noisyblock = mat_file['BenchmarkNoisyBlocksSrgb']

    # print(noisyblock)
    print('input shape', noisyblock.shape)

    transform = Compose([
        ToTensor()
    ])

    revtransform = Compose([
        ToImage()
    ])

    saveimgs = True    # if saveimgs is True, outputs of model will be saved as a .png

    # path of saved pkl file of model
    modelpath = f'checkpoints/swjung_track2.pkl'
    expname = 'swjung_track2'

    # get the name of pkl file
    filename = modelpath.split('/')[len(modelpath.split('/'))-1].split('.')[0]

    # set gpu
    device = torch.device('cuda:0')

    # make network object
    model = ntire_rdb_gd_rir_ver1(input_channel=3, numofrdb=20, numoffilters=64, t=1).to(device)

    # make numpy of output with same shape of input
    resultNP = np.ones(noisyblock.shape)
    print('resultNP.shape', resultNP.shape)

    # set paths of folders where images and output .mat file saved
    #'make_dirs' method makes a folder if there is no folder at path of input parameter
    submitpath = f'results_folder/{expname}'
    make_dirs(submitpath)
    imgsavepath = submitpath + '/imgs'
    make_dirs(imgsavepath)
    resultpath = submitpath
    make_dirs(resultpath)

    # load checkpoint of the model
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['state_dict'])

    # pass inputs through model and get outputs
    with torch.no_grad():

        model.eval()
        starttime = time.time()     # check when model starts to process
        for imgidx in tqdm.tqdm(range(noisyblock.shape[0])):
            for patchidx in range(noisyblock.shape[1]):
                img = noisyblock[imgidx][patchidx]   # img shape (256, 256, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # our model is optimized for BGR image so convert it

                input = transform(img)      # transform input numpy to tensor
                input = input.float().unsqueeze_(0).to(device)

                output = model(input)       # pass input through model

                outimg = revtransform(output)   # transform output tensor to numpy
                outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)    # revert BGR to RGB

                # put output patch into result numpy
                resultNP[imgidx][patchidx] = outimg

                if saveimgs:        # save output patches as .png file if saveimgs == True
                    # print('img shaved at', f'{imgsavepath}/{imgidx}_{patchidx}_{filename}_output.png')
                    cv2.imwrite(f'{imgsavepath}/{imgidx}_{patchidx}_{filename}_output.png', cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR))

    # check time after finishing task for all input patches
    endtime = time.time()
    elapsedTime = endtime - starttime   # calculate elapsed time
    print('ended', elapsedTime)
    num_of_pixels = noisyblock.shape[0] * noisyblock.shape[1] * noisyblock.shape[2] * noisyblock.shape[3]
    print('number of pixels', num_of_pixels)
    runtime_per_mega_pixels = (num_of_pixels / 1000000) / elapsedTime
    print('Runtime per mega pixel', runtime_per_mega_pixels)

    # save result numpy as .mat file
    sio.savemat(f'{resultpath}/{expname}', dict([('results', resultNP)]))
