#
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['image.cmap'] = 'gray'

caffe_root = os.environ.get("CAFFE_DIR") 

caffe.set_mode_cpu()

net = caffe.Net(caffe_root + '/examples/mnist/lenet.prototxt',
    caffe_root + '/examples/mnist/lenet_iter_10000.caffemodel',
    caffe.TEST)

# net = caffe.Net(caffe_root + '/models/bvlc_alexnet/deploy.prototxt',
#     caffe_root + '/models/bvlc_alexnet/bvlc_alexnet.caffemodel',
#     caffe.TEST)

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]),
                (0,padsize),
                (0,padsize)) + ( (0,0) , ) * (data.ndim -3)
    data = np.pad( data , padding, mode='constant',
                constant_values=(padval, padval))

    data = data.reshape( (n,n) + data.shape[1:]) \
                .transpose( (0,2,1,3) + tuple(range(4, data.ndim + 1)))

    data = data.reshape( (n * data.shape[1], n* data.shape[3]) # + data.shape[4:])
                )
    plt.imshow( data )

    plt.show()

print("net.paramsのkeyとvalueのリストを表示")
print([(k, v[0].data.shape) for k, v in net.params.items()])

filters = net.params['conv1'][0].data  # 20 images
vis_square(filters.transpose(0,2,3,1))

filters = net.params['conv2'][0].data  # 50*20 images 
vis_square(filters.reshape(50*20,5,5))
