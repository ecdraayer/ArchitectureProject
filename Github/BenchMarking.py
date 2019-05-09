"""Compare speed of different models with batch size 16"""
import torch
from torchvision.models import squeezenet, resnet, densenet, vgg, alexnet
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import time
import os
import numpy as np
import pandas as pd
import pandas
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt




def GraphResults(t):
 

    getpath =  os.path.join(os.getcwd(),'Graphs')
    if t == 'training':
        data = [x for x in glob.glob(getpath+'/*training*.csv')]
    elif t == 'testing':
        data = [x for x in glob.glob(getpath+'/*testing*.csv')]

    data.sort()

    c_model = data[0].split('/')[-1].split('_')[1]
    print(c_model)
    n = 4

    print(data[0])
    print(data[1])
    print(data[2])

    d = pd.read_csv(data[0])
    means_d = d.mean().values

    h = pd.read_csv(data[1])
    means_h =h.mean().values

    f = pd.read_csv(data[2])
    means_f =f.mean().values

    fig, ax = plt.subplots()

    index = np.arange(n)
    bar_width = 0.2

    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, means_d, bar_width,color='r',label='Double')
    rects2 = ax.bar(index + bar_width, means_f, bar_width,color='b',label='Single')
    rects2 = ax.bar(index + bar_width*2, means_h, bar_width,color='g',label='Half')

    
    ax.set_title(device_name+ " " + t + " Using " + str(Image_size)+'x'+str(Image_size) + " Images, Batch Size = " + str(batch_size))
    ax.set_xlabel('Convolutional Neural Network Type')
    ax.set_ylabel('Time(ms)')
    ax.legend()
    ax.set_xticks(index + 0.2)
    ax.set_xticklabels(f.columns,rotation=0, fontsize=7)
    fig.tight_layout()
    plt.savefig(device_name+'_'+t+'.png',dpi=400)

def create_model(p_type, model_type, c_model):
    model = getattr(model_type, c_model)(pretrained=False)

    if p_type is 'double':
        model=model.double()
    elif p_type is 'float':
        model=model.float()
    elif p_type is 'half':
        model=model.half()

    return model

def BenchMark_train():

    for p_type in precisions:
        print("Testing Precision: " + p_type)
        benchmarks = {}
        
        pictures = GetImages(p_type)
        final_layer = Variable(torch.LongTensor(batch_size).random_(Num_classes)).cuda()
        
        for models in CNNS.keys():
            for c_model in CNNS[models]:

                print('Training ' + c_model)
                deltas = []

                CNNmodel = create_model(p_type, models, c_model)
                CNNmodel.cuda()
                CNNmodel.train()
                cri = nn.CrossEntropyLoss()

                go = time.time()
                for e in range(epochs):
                    for s in range(0,Num_images,batch_size):
                        pictures = GetImages(p_type)

                        for t in range(Num_tests):      
                            #go = time.time()              
                            CNNmodel.zero_grad()
                            classification = CNNmodel.forward(pictures)
                            loss = cri(classification, final_layer)
                            loss.backward()
                            #stop = time.time()
                            #delta = (stop-go)*1000
                            #deltas.append(delta)
                        
                stop = time.time()
                print((stop-go)*1000)
                delta = (stop-go)*1000
                deltas.append(delta)
                del CNNmodel
                benchmarks[c_model] = deltas

        TRAIN_benchmark = pandas.DataFrame(benchmarks)
        TRAIN_benchmark.to_csv('Graphs/GPU_'+device_name+"_"+p_type+'_training_'+str(Image_size)+'x'+str(Image_size)+'_benchmark.csv', index=False)

def BenchMark_test():
    
    for p_type in precisions:

        benchmarks = {}
        pictures = GetImages(p_type)
        
        with torch.no_grad():
            for models in CNNS.keys():
                for c_model in CNNS[models]:

                    print('Testing ' + c_model)
                    deltas = []

                    CNNmodel = create_model(p_type,models,c_model)
                    CNNmodel.cuda()
                    CNNmodel.eval()
                    
                    go = time.time()
                    for e in range(epochs):
                        for s in range(0, Num_images, batch_size):

                            pictures = GetImages(p_type)
        
                            for t in range(Num_tests):
                                #go = time.time()
                                CNNmodel.forward(pictures)
                                #stop = time.time()
                                #delta = (stop-go)*1000
                                #deltas.append(delta)
                    stop = time.time()
                    print((stop-go)*1000)  
                    delta = (stop-go)*1000 
                    deltas.append(delta)   
                    del CNNmodel
                    benchmarks[c_model] = deltas

        TEST_benchmark = pandas.DataFrame(benchmarks)
        TEST_benchmark.to_csv('Graphs/GPU_'+device_name+"_"+p_type+'_testing_'+str(Image_size)+'x'+str(Image_size)+'_benchmark.csv', index=False)


def GetImages(p_type):
    pictures = Variable(torch.randn(batch_size,
                               3, 
                               Image_size, 
                               Image_size), 
                               requires_grad=True).cuda()

    if p_type is 'double':
        pictures=pictures.double()
    elif p_type is 'float':
        pictures=pictures.float()
    else:
        pictures=pictures.half()

    return pictures

device_name=torch.cuda.get_device_name(0)

CNNS = {
    vgg: ['vgg16'],
    resnet:['resnet50'],
    densenet: ['densenet121'],
    squeezenet: ['squeezenet1_1']
}

precisions=['double','half','float']
#precisions=['float']
#precisions=['half']
#precisions=['double']

batch_size = 5
Num_tests = 30
Num_classes = 10
Image_size = 256
epochs = 1
Num_images = 5

torch.backends.cudnn.benchmark = True
BatchSizes = [ [[4500, 1250, 325, 85, 20],[3100,875,225,55,14],[31250,8250,1600,375,85],[3100,1100,300,75,20]],
               [[9000,2500,650,165,40],[6200,1800,450,110,25],[560000,15000,3000,650,150],[6250,1700,600,150,35]],
               [[2250,650,160,40,10],[1550,400,110,25,7],[16500,4500,900,200,40],[1300,450,125,30,7]]]


ImageSizes = [16,32,64,128,256]

if __name__ == '__main__':
    os.makedirs('Graphs', exist_ok=True)
    print('GPU:', torch.cuda.get_device_name(0))
    print('Batch Size: ', batch_size)
    print('Image_size: ', Image_size)
    print('Num_classes ', Num_classes)
    print('Num_tests ', Num_tests)
    #for size in BS:
    torch.cuda.empty_cache()
    BenchMark_train()
    torch.cuda.empty_cache()
    BenchMark_test()
    
    GraphResults('training')
    GraphResults('testing')