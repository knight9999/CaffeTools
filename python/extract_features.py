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

def vis_square(name, data, padsize=1, padval=0):
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

    fig = plt.gcf()
    plt.imshow( data )

    plt.show()
    fig.savefig( "result/" + name + "_data.png", dpi=100)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))

# transformer.set_mean('data',np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))

transformer.set_raw_scale('data', 255)

net.blobs['data'].reshape(1,1,28,28)

# net.blobs['data'].data[...] = transformer.preprocess('data',
#     caffe.io.load_image(caffe_root + '/examples/images/cat_gray.jpg', color=False) )
net.blobs['data'].data[...] = transformer.preprocess('data',
    caffe.io.load_image(caffe_root + '/examples/images/no7.png', color=False) )
out = net.forward()

print("net.blobsのkeyとvlueのリストを表示")
print([ (k, v.data.shape) for k,v in net.blobs.items()])

features = net.blobs['conv1'].data[0, :20]
vis_square('conv1',features, padval=1)

features = net.blobs['conv2'].data[0, :50]
vis_square('conv2',features, padval=1)

features = net.blobs['ip2'].data
print ( features )

features = net.blobs['prob'].data
print ( features )
