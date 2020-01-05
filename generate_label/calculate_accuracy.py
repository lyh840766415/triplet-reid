#-*-coding:GBK -*-
import tensorflow as tf
import numpy as np
import random

batch = 10000
palce_num = 5

same_label = tf.placeholder(tf.int32,shape=[batch,batch])
same_label = tf.cast(same_label,tf.bool)

#input = tf.constant(np.random.rand(batch,batch))

input_placeholder = tf.placeholder(tf.float32,shape=[batch,batch])
value,indices = tf.nn.top_k(input_placeholder, 5)
batch_index = tf.tile(
    tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
    (1, tf.shape(indices)[1]))
    
#topk_indices = (batch_index, indices)
topk_indices = tf.stack((batch_index, indices), -1)
topk_is_same = tf.gather_nd(same_label, topk_indices)
topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
top1 = tf.reduce_mean(topk_is_same_f32[:,0])
prec_at_k = tf.reduce_mean(topk_is_same_f32)




	
			
#indices = tf.nn.sort(input)
#indices = indices[:,1:]
label = np.empty([batch,batch],np.int32)
for i in range(batch):
	for j in range(batch):
		k = i//palce_num
		if j<(k+1)*palce_num and j>=k*palce_num:
			label[i,j]=1
		else:
			label[i,j]=0

print(label)

input = np.empty([batch,batch],np.float32)
for i in range(batch):
	for j in range(batch):
			input[i,j]=random.randint(1,99)/100.0

print(input)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
	'''
	print(sess.run(input))
	print(sess.run(indices))
	print(sess.run(top1))
	print(sess.run(top4))
	'''
	#ind,run_input,run_value,run_batch_index,run_topk_indices,run_topk_is_same,run_same_label,run_topk_is_same_f32,run_top1,run_prec_at_k = sess.run([indices,input_placeholder,value,batch_index,topk_indices,topk_is_same,same_label,topk_is_same_f32,top1,prec_at_k],feed_dict={same_label:label,input_placeholder:input})
	run_top1,run_prec_at_k = sess.run([top1,prec_at_k],feed_dict={same_label:label,input_placeholder:input})


	print(run_top1)
	print(run_prec_at_k)