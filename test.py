import numpy as np
import cv2
import torch
from models.DenoisingModels import *
import tqdm
from utils.utils import *
from utils.transforms import *
import scipy.io as sio
import time
from PIL import Image

if __name__ == '__main__':

    transform = Compose([
        ToTensor()
    ])

    revtransform = Compose([
        ToImage()
    ])

    # Read .mat file
    #mat_file = sio.loadmat('B_3.mat')
    # get input numpy
    #noisyblock = mat_file['img_noisy']
    #cc = Image.fromarray(noisyblock)    # 不转为uint8的话，Image.fromarray这句会报错
    #cc.save('B_3.png')

    # path of saved pkl file of model
    modelpath = f'checkpoints/DGU-3DMlab1_track2.pkl'
    expname = 'DGU-3DMlab1_track2'

    # get the name of pkl file
    filename = modelpath.split('/')[len(modelpath.split('/'))-1].split('.')[0]

    # set gpu
    device = torch.device('cuda:0')

    # make network object
    model = ntire_rdb_gd_rir_ver2(input_channel=3, numoffilters=80, t=1).to(device)

    # set paths of folders where images and output .mat file saved
    #'make_dirs' method makes a folder if there is no folder at path of input parameter
    submitpath = f'results_folder/{expname}'
    make_dirs(submitpath)
    imgsavepath = submitpath + '/imgs'
    make_dirs(imgsavepath)
    resultpath = submitpath

    # load checkpoint of the model
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['state_dict'])

    # pass inputs through model and get outputs
    with torch.no_grad():

        model.eval()
        img = np.array(Image.open('001.png'))
        starttime = time.time()     # check when model starts to process
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # our model is optimized for BGR image so convert it

        input = transform(img)      # transform input numpy to tensor
        input = input.float().unsqueeze_(0).to(device)

        output = model(input)       # pass input through model

        outimg = revtransform(output)   # transform output tensor to numpy
        outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)    # revert BGR to RGB
        z = Image.fromarray(np.uint8(outimg))    # 不转为uint8的话，Image.fromarray这句会报错
        z.save('001_result.png')

    # check time after finishing task for all input patches
    endtime = time.time()
    elapsedTime = endtime - starttime   # calculate elapsed time
    print('ended', elapsedTime)
