#-*-coding:GBK -*-
import numpy as np
import tensorflow as tf
import random
import os
import shutil
import common
import cv2
import nets.resnet_v1_50 as resnet
import heads.fc1024 as header
import loss

def select_training_batches(LABEL_PATH,seq_name,DATA_PATH,place_num,pos_num):
	if not os.path.exists(os.path.join(LABEL_PATH,seq_name,"margin%s.txt"%(seq_name))):
		print("error: file not exist")
	
	margin_mat = np.loadtxt(os.path.join(LABEL_PATH,seq_name,"margin%s.txt"%(seq_name)))
	
	img_num = margin_mat.shape[0]
	#print(margin_mat.shape[0])
		
	#sort the margin_mat to find the appropriate inter_dis & intra_dis
	margin_array = np.sort(margin_mat.flatten())
	intra_dis = margin_array[img_num*img_num//50]
	inter_dis = margin_array[img_num*img_num//5]
	

	batch_num = img_num//(place_num*pos_num)



	failed_batch = 0
	batch_sample = np.zeros([batch_num,place_num,pos_num],dtype=np.int)
	
	for i in range(batch_num):
		#print("batch %d"%(i))
		
		#init mask for cur batch
		failed = True
		while(failed):
			#print("failed_batch %d"%(i))
			failed = False
			mask = np.arange(0,img_num,1)
			for j in range(place_num):
				#random select 1 positive		
				if(mask.shape[0]<=place_num):
					#print("error, lack of anchor")
					failed_batch+=1;
					failed = True
					break
					
				pos = random.randint(0,mask.shape[0]-1)
				batch_sample[i,j,0] = mask[pos]
				
				#print("place %d"%(j))
				#print(pos)
				
				#init pos_k_list
				pos_list = np.empty(0,dtype = int)
				del_list = np.empty(0)
				del_list = np.append(del_list,pos)
							
				#set impossible point mast to 0
				for l in range(mask.shape[0]):
					if(margin_mat[mask[pos],mask[l]]<intra_dis):
						#append to pos_k_list
						pos_list = np.append(pos_list,mask[l])
						del_list = np.append(del_list,l)
					elif(margin_mat[mask[pos],mask[l]]<inter_dis):
						del_list = np.append(del_list,l)
				
				mask = np.delete(mask,del_list)
				
				if(pos_list.shape[0]<pos_num-1):
					#print("error, lack of positive")
					failed_batch+=1;
					failed = True
					break
				for k in range(pos_num-1):
					
					#flag of ensure distance
					fake = True		
					fake_time = 0;
					while fake:
						#print("fake")
						if pos_list.shape[0]<1 or fake_time>2*pos_list.shape[0]:
							#print("error, lack of positive")
							failed_batch+=1;
							failed = True
							break
							
						fake = False				
						pos_k = random.randint(0,pos_list.shape[0]-1)				
						#ensure that distance between one place
						for k1 in range(k+1):						
							if(margin_mat[pos_list[pos_k],batch_sample[i,j,k1]] > intra_dis):
								fake = True
								fake_time+=1
								
						
					batch_sample[i,j,k+1] = pos_list[pos_k]
					pos_list = np.delete(pos_list,pos_k)
					#select k positive
		#print(batch_sample[i,:,:])
		
	#print("batch select finish")
	
	batch_sample = batch_sample.flatten()
	#print(batch_sample)
	fids = np.empty(0,dtype=object)
	pids = np.empty([batch_sample.shape[0],2],dtype=np.uint16)
	for i in range(batch_sample.shape[0]):
		image_name = os.path.join(DATA_PATH,seq_name,seq_name[0:10],seq_name,"image_02/data","%010d.png"%(batch_sample[i]))
		fids = np.append(fids,image_name)
		pids[i,0]=i
		pids[i,1]=batch_sample[i]
		#print(image_name)
		if not os.path.exists(image_name):
			print("error: image not exist")
		
	return fids,pids,batch_num,margin_mat
	
'''
This function is used to select sample for test
Input:
	LABEL_PATH: PATH for the margin matrix
	seq_dirs: all sequence directions
	seq_num: test sequence number
	DATA_PATH: PATH that contains IMAGE
	place_num: selected place per sequence
	pos_num: selected positive per place
Output
	fids: FILE PATH for the selected image
	pids: PLACE ID in each sequence
'''
def select_testing_batches(LABEL_PATH,seq_dirs,seq_num,DATA_PATH,place_num,pos_num):
	#count the image number in each data directories, use this to allocate probability
	tot_image_num = 0;
	tot_seq_num = len(seq_dirs)
	seq_image_size = np.empty(tot_seq_num,dtype=np.int)
	acc_seq_image_size = np.empty(tot_seq_num,dtype=np.int)
	has_selected = np.zeros(tot_seq_num,np.bool)
	
	for i,seq_name in enumerate(seq_dirs):
		file_path = os.path.join(DATA_PATH,seq_name,seq_name[0:10],seq_name,"image_02/data")
		seq_image_size[i] = len(os.listdir(file_path))
		if i == 0:
			acc_seq_image_size[0] = seq_image_size[0]
		else:
			acc_seq_image_size[i] = acc_seq_image_size[i-1]+seq_image_size[i]
		#print(len(os.listdir(file_path)))
		#print(acc_seq_image_size[i])
		#print(seq_image_size[i])
	
	
		
	tot_image_num = np.sum(seq_image_size)
	#print("training set size = %d"%(tot_image_num))

	#select the batches
	fids = np.empty(seq_num*place_num*pos_num,dtype=object)
	pids = np.empty([seq_num*place_num*pos_num,2],dtype=np.int)
	batch_sample = np.zeros([seq_num,place_num,pos_num],dtype=np.int)
	#init select_seq
	select_seq = 0
	
	for i in range(seq_num):
		seq_select_failed = True
		while(seq_select_failed):
			#generate a random number
			selected = random.randint(0,tot_image_num)
			#decide the seq
			for j in range(tot_seq_num):
				if selected > acc_seq_image_size[j]:
					continue
				else:
					select_seq = j
					if not has_selected[select_seq]:
						print("select_seq = %d"%(select_seq))
						has_selected[select_seq] = True
						seq_select_failed = False
						break
		
		print("loading matrix %d x %d"%(seq_image_size[select_seq],seq_image_size[select_seq]))
		margin_mat = np.loadtxt(os.path.join(LABEL_PATH,seq_dirs[select_seq],"margin%s.txt"%(seq_dirs[select_seq])))
		margin_array = np.sort(margin_mat.flatten())
		#intra_ind = max(10000,seq_image_size[select_seq]*seq_image_size[select_seq]//100)
		#intra_dis = margin_array[1000]
		intra_dis = margin_array[seq_image_size[select_seq]*seq_image_size[select_seq]//50]
		
		inter_dis = margin_array[seq_image_size[select_seq]*seq_image_size[select_seq]//5]
		
		failed = True
		while(failed):
			#print("failed_batch %d"%(i))
			failed = False
			mask = np.arange(0,seq_image_size[select_seq],1)
			for j in range(place_num):
				#random select 1 positive		
				if(mask.shape[0]<=place_num):
					#print("error, lack of anchor")
					failed = True
					break
					
				pos = random.randint(0,mask.shape[0]-1)
				batch_sample[i,j,0] = mask[pos]
				
				
				#init pos_k_list
				pos_list = np.empty(0,dtype = int)
				del_list = np.empty(0)
				del_list = np.append(del_list,pos)
							
				#set impossible point mast to 0
				for l in range(mask.shape[0]):
					if(margin_mat[mask[pos],mask[l]]<intra_dis and margin_mat[mask[pos],mask[l]]>0):
					#if(margin_mat[mask[pos],mask[l]]<intra_dis and mask[pos] != mask[l]):
						#append to pos_k_list
						pos_list = np.append(pos_list,mask[l])
						del_list = np.append(del_list,l)
					elif(margin_mat[mask[pos],mask[l]]<inter_dis):
						del_list = np.append(del_list,l)
				
				mask = np.delete(mask,del_list)
				
				if(pos_list.shape[0]<pos_num-1):
					#print("error, lack of positive")
					failed = True
					break
				for k in range(pos_num-1):
					
					#flag of ensure distance
					fake = True		
					fake_time = 0;
					while fake:
						#print("fake")
						if pos_list.shape[0]<1 or fake_time>2*pos_list.shape[0]:
							#print("error, lack of positive")
							failed = True
							break
							
						fake = False				
						pos_k = random.randint(0,pos_list.shape[0]-1)				
						#ensure that distance between one place
						for k1 in range(k+1):						
							if(margin_mat[pos_list[pos_k],batch_sample[i,j,k1]] > intra_dis):
								fake = True
								fake_time+=1
								
						
					batch_sample[i,j,k+1] = pos_list[pos_k]
					pos_list = np.delete(pos_list,pos_k)
					
		#validate the margin between selected samples
		for j in range(place_num):
			for k in range(place_num):
				if margin_mat[batch_sample[i][j][0],batch_sample[i][k][0]]<inter_dis and j!=k:
					print("inter sample error")
			for k in range(pos_num):
				for k1 in range(pos_num):
					if k == k1:
						continue
					if margin_mat[batch_sample[i][j][k],batch_sample[i][j][k1]]>intra_dis:
						print("intra sample error")
		
		for j in range(place_num):
			for k in range(pos_num):
				image_name = os.path.join(DATA_PATH,seq_dirs[select_seq],seq_dirs[select_seq][0:10],seq_dirs[select_seq],"image_02/data","%010d.png"%(batch_sample[i][j][k]))
				
				cur_ind = i*place_num*pos_num+j*pos_num+k
				#print(cur_ind)
				fids[cur_ind] = image_name
				pids[cur_ind] = cur_ind
				pids[cur_ind,1] = batch_sample[i][j][k]
	
	'''		
	for i in range(fids.shape[0]):
		print(fids[i])
	#	print(pids[i])
	'''
		
	return fids,pids
	
def match_fid_pid(pid,fids):
	return pid[1],fids[pid[0]]
	
def fid_to_image(pid,fid,image_size):
	image_encoded = tf.read_file(fid)
	image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
	image_resized = tf.image.resize_images(image_decoded, image_size)
	return image_resized,pid,fid
	
def init_dataset_loader(batch,id,path,batch_size):
	loading_threads = 8
	net_input_size = (144, 288)
	dataset = tf.data.Dataset.from_tensor_slices(id)
	dataset = dataset.map(lambda pid: match_fid_pid(pid,path))
	dataset = dataset.map(lambda pid,fid: fid_to_image(pid,fid,image_size=net_input_size),num_parallel_calls=loading_threads)
	dataset = dataset.batch(batch_size[0])
	iter = dataset.make_initializable_iterator()
	images,pids,fids = iter.get_next()
	return images,pids,fids,iter

def init_network(images,embedding_dim,margin,same_label_ph,batch):
	endpoints,body_prefix=resnet.endpoints(images,is_training=True)
	endpoints = header.head(endpoints,embedding_dim,is_training=True)
	
	#init the loss function and compute the loss
	dists = loss.cdist(endpoints['emb'], endpoints['emb'], metric="euclidean")
	losses = loss.batch_loss(margin,dists)
	#compute the accuracy and recall
	accuracies,recalles = loss.batch_accuracy_recall(dists,same_label_ph,batch)
	

	loss_mean = tf.reduce_mean(losses)
	accuracy_mean = tf.reduce_mean(accuracies)
	recall_mean = tf.reduce_mean(recalles)
	
	#summary the loss/accuracy/recall
	summary_loss = tf.summary.scalar('loss',loss_mean)
	summary_accuracy = tf.summary.scalar('accuracy',accuracy_mean)
	summary_recall = tf.summary.scalar('recall',recall_mean)
	
	
	#optimization
	#all tensorflow variable will send to the GLOBAL_VARIBALES and the "body_prefix" is a string that use to filter variables
	#just used to save as record point
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)
	
	#learning rate strategy
	global_step = tf.Variable(0, name='global_step', trainable=False)
	#boundaries = [3500,7000,10500,14000,17500,21000,24500,28000]
	#rates = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10.0]
	#learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=rates)
	learning_rate = tf.train.exponential_decay(1e-3,global_step,3500,0.1)
	summary_learning_rate = tf.summary.scalar('learning_rate',learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		train_op = optimizer.minimize(loss_mean, global_step=global_step)
	
	return train_op,global_step,losses,accuracy_mean,recall_mean,summary_loss,summary_accuracy,summary_recall,summary_learning_rate
	


	for i,seq_name in enumerate(seq_dirs):
		file_path = os.path.join(DATA_PATH,seq_name,seq_name[0:10],seq_name,"image_02/data")
		seq_image_size[i] = len(os.listdir(file_path))
		if i == 0:
			acc_seq_image_size[0] = seq_image_size[0]
		else:
			acc_seq_image_size[i] = acc_seq_image_size[i-1]+seq_image_size[i]
		#print(len(os.listdir(file_path)))
		#print(acc_seq_image_size[i])
		print(seq_image_size[i])


def main():
	batch_p=5
	batch_k=5
	test_seq_num = 4
	batch_size = batch_k*batch_p
	test_batch_size = np.empty(1,dtype = np.int)
	test_batch_size[0] = test_seq_num*batch_k*batch_p
	train_batch_size = np.empty(1,dtype = np.int)
	train_batch_size[0] = batch_k*batch_p
	epoch = 100
	seq_num = 0
	embedding_dim = 128
	LABEL_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/data"
	DATA_PATH = "/home/luyuheng/lab/dataset/KITTI_RAW"
	LOG_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/experiment"
	
	dirs = os.listdir(LABEL_PATH)
	seq_dirs=np.empty(0,dtype=object)
	for cur_dir in dirs :
		if os.path.isdir(os.path.join(LABEL_PATH,cur_dir)):
			#保证序列长度
			if len(os.listdir(os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data"))) > 1000:
				seq_num+=1	
				seq_dirs = np.append(seq_dirs,cur_dir)
	
	'''
	#for debug		
	select_testing_batches(LABEL_PATH,seq_dirs,test_seq_num,DATA_PATH,batch_p,batch_k)
	exit()
	'''
			
	#print(seq_dirs)
	
	#init of read of dataset
	x = tf.placeholder(tf.int32,shape=[None,2])
	y = tf.placeholder(tf.string,shape=[None])
	batch_per_run = tf.placeholder(tf.int64,shape=[1])
	images,pids,fids,iter = init_dataset_loader(batch_size,x,y,batch_per_run)
	
	#init the neural network model
	#TODO 2019_11_20 Use resnet model
	same_label = np.empty([test_batch_size[0],test_batch_size[0]],np.int32)
	for i in range(test_batch_size[0]):
		for j in range(test_batch_size[0]):
			k = i//batch_p
			if j<(k+1)*batch_p and j>=k*batch_p:
				same_label[i,j]=1
			else:
				same_label[i,j]=0			
	margin = tf.placeholder(tf.float32,shape = [batch_size, batch_size])
	same_label_ph = tf.placeholder(tf.int32,shape = [test_batch_size[0], test_batch_size[0]])
	train_op,global_step,losses,accuracy_mean,recall_mean,summary_loss,summary_accuracy,summary_recall,summary_learning_rate = init_network(images,embedding_dim,margin,same_label_ph,test_batch_size[0])
	print(same_label)
	
	#used to save the labeled margin matrix
	label_margin = np.empty([batch_size,batch_size], dtype = np.float32)
	#main loop
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		#initial all model variables
		sess.run(tf.global_variables_initializer())
		#merged_summary = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(LOG_PATH,sess.graph)
		step = 0		
		
		for ep in range(epoch):
			#testing
			#select the testing ID and testing FILES
			
			_fids,_pids = select_testing_batches(LABEL_PATH,seq_dirs,test_seq_num,DATA_PATH,batch_p,batch_k)
			#send the test batch to the network
			sess.run(iter.initializer,feed_dict={x:_pids,y:_fids,batch_per_run:test_batch_size})
			#print(sess.run(fids))
			
			#set the network batch size changeable
			#if memory is a problem, then calculate the accuracy and recall in the numpy
			test_accuracy,test_recall,summary_acc,summary_rc= sess.run([accuracy_mean,recall_mean,summary_accuracy,summary_recall],feed_dict={same_label_ph:same_label})
			summary_writer.add_summary(summary_acc,step)
			summary_writer.add_summary(summary_rc,step)
			print("iter:%6d, accuracy = %.3f, recall = %.3f"%(step,test_accuracy,test_recall))
			
			
			#training
			for seq in range(seq_num):
				#print("seq_num = %d path = %s"%(seq,seq_dirs[seq]))
				_fids, _pids, _bat_per_seq, _margin_mat = select_training_batches(LABEL_PATH,seq_dirs[seq],DATA_PATH,batch_p,batch_k)
				sess.run(iter.initializer,feed_dict={x:_pids,y:_fids,batch_per_run:train_batch_size})
					
				for bat in range(_bat_per_seq):
					print("epoch = %d, seq = %d, bat = %d"%(ep,seq,bat))
					#generate the label margin_matrix
					bat_start=bat*batch_size
					for col in range(batch_size):
						for row in range(batch_size):
							col_sample = _pids[bat_start+col,1]
							row_sample = _pids[bat_start+row,1]
							label_margin[col,row] = _margin_mat[col_sample,row_sample]
						
					_, step, b_loss,summary_ls,summary_lr = sess.run([train_op,global_step,losses,summary_loss,summary_learning_rate],feed_dict={margin:label_margin})
					
					summary_writer.add_summary(summary_ls,step)
					summary_writer.add_summary(summary_lr,step)
						
					print("iter:{%6d}, loss min|avg|max: {%.3f}|{%.3f}|{%6.3f})"%(step,np.min(b_loss),np.mean(b_loss),np.max(b_loss)))

if __name__ == '__main__':
	main()
