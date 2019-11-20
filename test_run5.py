# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:55:12 2019

@author: yorksywang
"""

import tensorflow as tf
import random 
import numpy as np
saver='test'
part_id=1
count=1
save_name=saver+str(part_id)
writer = tf.python_io.TFRecordWriter(save_name)
for i in range(100000):
    trade_f=np.random.rand(416,1).reshape(-1)
    pos_t=np.random.rand(768,1).reshape(-1)
    neg_t=np.random.rand(768,1).reshape(-1)
    example = tf.train.Example( #We need example format to write into tfrecords files
                                features =
                tf.train.Features(feature={
                "trade_features": tf.train.Feature(bytes_list  =tf.train.BytesList(value=[trade_f.astype(np.float32).tostring()])),
                "pos_text": tf.train.Feature(bytes_list  =tf.train.BytesList(value=[pos_t.astype(np.float32).tostring()])),
                "neg_text": tf.train.Feature(bytes_list  =tf.train.BytesList(value=[neg_t.astype(np.float32).tostring()])),
            }
    )
            )
    serialized = example.SerializeToString()
    writer.write(serialized)
    count+=1
    if count%50000==0:
                part_id+=1
                save_name=saver+str(part_id)
                writer.close()
                writer = tf.python_io.TFRecordWriter(save_name)
writer.close()



import tensorflow as tf
from tensorflow.contrib import slim

data_file = "./test1"
reader = tf.TFRecordReader

keys_to_features = {
	#we should specify the shape of array to restore
    'trade_features': tf.io.FixedLenFeature([], dtype=tf.string), 
             
            # vector的shape刻意从原本的(3,)指定成(1,3)
    'pos_text': tf.io.FixedLenFeature([], dtype=tf.string), 
            
            # 使用 VarLenFeature来解析
    'neg_text': tf.io.FixedLenFeature([], dtype=tf.string)
}

items_to_handlers = {
    'trade_features' : slim.tfexample_decoder.Tensor('trade_features'),
    'pos_text' : slim.tfexample_decoder.Tensor('pos_text'),
    'neg_text' : slim.tfexample_decoder.Tensor('neg_text')
}

items_to_descriptions = {
    'trade_features' : 'a 416 int64 array',
    'pos_text' : 'a 768 float32 array',
    'neg_text' : 'a 768 float32 array'
}
decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                  items_to_handlers)

dataset = slim.dataset.Dataset(data_sources=data_file, reader=reader,
                               decoder=decoder, num_samples=2,
                               items_to_descriptions=items_to_descriptions)

provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset, num_readers=1, common_queue_capacity=100, common_queue_min=4)

[trade_features, pos_text, neg_text] = provider.get(['trade_features', 'pos_text', 'neg_text'])


b_iarray, b_farray,bb = tf.train.batch([trade_features, pos_text, neg_text], batch_size=1,
                                    num_threads=1, capacity=5)
b_iarray=tf.decode_raw(b_iarray,tf.float32)
b_iarray=tf.reshape(b_iarray,[416])
#b_iarray=b_iarray.reshape(-1)
b_farray=tf.decode_raw(b_farray,tf.float32)
b_farray=tf.reshape(b_farray,[768])
#b_farray=tb_farray.reshape(-1)
bb=tf.decode_raw(bb,tf.float32)
bb=tf.reshape(bb,[768])
#bb=bb.reshape(-1)
batch_queue = slim.prefetch_queue.prefetch_queue([b_iarray, b_farray,bb],
                                                 capacity=1)

sess = tf.Session()
thread = tf.train.start_queue_runners(sess=sess)

iarray, farray,b = batch_queue.dequeue()
ia, fa,fb = sess.run([iarray, farray,b])

print(ia[0])
print(fa[0])