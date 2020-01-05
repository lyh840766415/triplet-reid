#-*-coding:GBK -*-
import numpy as np
import tensorflow as tf
import random
import os
import nets.resnet_v1_50 as resnet
import time
import loss

#read all margin file
#Global variable: save all margin
margin_mats = []

def load_sequence(LABEL_PATH,DATA_PATH):
	seq_num = 0
	dirs = os.listdir(LABEL_PATH)
	seq_dirs=np.empty(0,dtype=object)
	for cur_dir in dirs :
		if os.path.isdir(os.path.join(LABEL_PATH,cur_dir)):
			#保证序列长度
			if len(os.listdir(os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data"))) > 300:
				seq_num+=1	
				seq_dirs = np.append(seq_dirs,cur_dir)
	return seq_dirs,seq_num
	
def load_margin(LABEL_PATH,seq_dirs):
	for seq_name in seq_dirs:
		print("loading %s"%(seq_name))
		margin_mat = np.loadtxt(os.path.join(LABEL_PATH,seq_name,"margin%s.txt"%(seq_name)))
		margin_mats.append(margin_mat)

def select_batch(DATA_PATH,seq_dirs,seq_per_batch,place_num,pos_num):

	#fellowing code is used to select sequence
	#count the image number in each data directories, use this to allocate probability
	tot_image_num = 0;
	tot_seq_num = len(seq_dirs)
	seq_image_size = np.empty(tot_seq_num,dtype=np.int)
	acc_seq_image_size = np.empty(tot_seq_num,dtype=np.int)
	#count the image number of each sequence
	for i,seq_name in enumerate(seq_dirs):
		file_path = os.path.join(DATA_PATH,seq_name,seq_name[0:10],seq_name,"image_02/data")
		seq_image_size[i] = len(os.listdir(file_path))
		if i == 0:
			acc_seq_image_size[0] = seq_image_size[0]
		else:
			acc_seq_image_size[i] = acc_seq_image_size[i-1]+seq_image_size[i]
	tot_image_num = np.sum(seq_image_size)
	#print(tot_image_num)
	
	batch_filename = np.empty(seq_per_batch*place_num*pos_num,dtype=object)
	batch_label = np.empty([seq_per_batch*place_num*pos_num,2],dtype=np.int)	
	batch_sample = np.zeros([seq_per_batch,place_num,pos_num],dtype=np.int)
	#select sequence
	seq_has_selected = np.zeros(tot_seq_num,np.bool)
	for i in range(seq_per_batch):
		seq_select_failed = True
		while(seq_select_failed):
			#generate a random number
			selected = random.randint(0,tot_image_num-1)
			#decide the seq
			for j in range(tot_seq_num):
				if selected > acc_seq_image_size[j]:
					continue
				else:
					select_seq = j
					if not seq_has_selected[select_seq]:
						print("select_seq = %d"%(select_seq))
						seq_has_selected[select_seq] = True
						seq_select_failed = False
						break
						
		#select place
		#args
		intra_dis_per = int(min(50,seq_image_size[select_seq]//20))
		#inter_dis_per = 25
		inter_dis_per = int(max(10,seq_image_size[select_seq]*-0.02+30))
		margin_mat = margin_mats[select_seq]
		margin_array = np.sort(margin_mat.flatten())
		intra_dis = margin_array[seq_image_size[select_seq]*seq_image_size[select_seq]//intra_dis_per]
		inter_dis = margin_array[seq_image_size[select_seq]*seq_image_size[select_seq]//inter_dis_per]
				
		place_select_failed = True
		while(place_select_failed):
			#print("failed_batch %d"%(i))
			place_select_failed = False
			mask = np.arange(0,seq_image_size[select_seq],1)
			for j in range(place_num):
				#random select 1 positive		
				if(mask.shape[0]<=place_num):
					#print("error, lack of anchor")
					place_select_failed = True
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
					place_select_failed = True
					break
					
				#select k positive	
				for k in range(pos_num-1):
					#flag of ensure distance
					fake = True		
					fake_time = 0;
					while fake:
						#print("fake")
						if pos_list.shape[0]<1 or fake_time>2*pos_list.shape[0]:
							#print("error, lack of positive")
							place_select_failed = True
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
				batch_filename[cur_ind] = image_name
				batch_label[cur_ind] = cur_ind
				#batch_label[cur_ind,1] = batch_sample[i][j][k]
				#batch_label[cur_ind,1] = select_seq
				batch_label[cur_ind,1] = i*place_num+j
				
	return batch_filename,batch_label
	

def match_fid_label(label,fids):
	return label[1],fids[label[0]]
	
def fid_to_image(pid,fid,image_size):
	image_encoded = tf.read_file(fid)
	image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
	image_resized = tf.image.resize_images(image_decoded, image_size)
	return image_resized,pid,fid

'''
init the dataset loader
Output:
	images: batch of images
	labels: label of images
'''
def init_dataset_loader(input_label,path,batch_size):
	loading_threads = 8
	net_input_size = (144, 288)
	dataset = tf.data.Dataset.from_tensor_slices(input_label)
	dataset = dataset.map(lambda label: match_fid_label(label,path))
	dataset = dataset.map(lambda label,fid: fid_to_image(label,fid,image_size=net_input_size),num_parallel_calls=loading_threads)
	dataset = dataset.batch(batch_size[0])
	iter = dataset.make_initializable_iterator()
	images,labels,fids = iter.get_next()
	return images,labels,fids,iter
	
def init_network(images,labels,embbed_size):
	endpoints,body_prefix=resnet.endpoints(images,is_training=True)
	
	#fix output feature to 128-d
	logits = tf.layers.dense(endpoints['model_output'], embbed_size)

	with tf.name_scope('loss'):
		#class_label = tf.one_hot(labels, num_classes)
		#losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, num_classes),logits=logits)
		dists = loss.cdist(logits, logits, metric="euclidean")
		losses,_,acces,_,_,_ = loss.batch_hard(dists,labels,"soft",1)
		
		mean_loss = tf.reduce_mean(losses)
		summary_loss = tf.summary.scalar('loss',mean_loss)
		
		mean_acc = tf.reduce_mean(acces)
		summary_acc = tf.summary.scalar('accuracy',mean_acc)


	train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(losses)
	return mean_loss,train_op,summary_loss,mean_acc,summary_acc


def main():
	#arguments
	seq_per_batch = 4
	batch_k = 5
	batch_p = 5
	batch_size = np.empty(1,dtype = np.int)
	batch_size[0] = seq_per_batch*batch_k*batch_p
	seq_num = 0	
	iter_num = 10000
	embbed_size = 128
	LABEL_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/data"
	DATA_PATH = "/home/luyuheng/lab/dataset/KITTI_RAW"
	LOG_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/experiment3"
	
	
	#load the validate sequence
	seq_dirs,seq_num = load_sequence(LABEL_PATH,DATA_PATH)
	
	#read all margin file to margin_mats
	load_margin(LABEL_PATH,seq_dirs)
	
	#init dataset loadder
	label_placeholder = tf.placeholder(tf.int32,shape=[None,2])
	filepath_placeholder = tf.placeholder(tf.string,shape=[None])
	batch_per_run_placeholder = tf.placeholder(tf.int64,shape=[1])
	images,labels,fids,iter = init_dataset_loader(label_placeholder,filepath_placeholder,batch_per_run_placeholder)
	
	#init network
	mean_loss,train_op,summary_loss,mean_acc,summary_acc = init_network(images,labels,embbed_size)
	
	#config GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	saver = tf.train.Saver()
	
	#Start training
	with tf.Session(config=config) as sess:
		summary_writer = tf.summary.FileWriter(LOG_PATH,sess.graph)
		sess.run(tf.global_variables_initializer())
		
		for step in range(iter_num):
			batch_filename,batch_label = select_batch(DATA_PATH,seq_dirs,seq_per_batch,batch_p,batch_k)
			train_feed_dict = {
				label_placeholder: batch_label,
				filepath_placeholder: batch_filename,
				batch_per_run_placeholder:batch_size
			}
			
			sess.run(iter.initializer,feed_dict=train_feed_dict)
			_,summary_loss_run,mean_loss_run,summary_acc_run,mean_acc_run = sess.run([train_op,summary_loss,mean_loss,summary_acc,mean_acc])
			summary_writer.add_summary(summary_loss_run,step)
			summary_writer.add_summary(summary_acc_run,step)
			
			if step%1000 == 0:
				saver.save(sess, os.path.join(LOG_PATH,"Model/batch_resnet_triplet_accuracy_iter_%06d_model.ckpt"%(step)))
			
			print("iter = %d, loss = %f, accuracy = %f"%(step,mean_loss_run,mean_acc_run))
			print(time.strftime('%H:%M:%S',time.localtime(time.time())))
		
	
if __name__ == '__main__':
	main()




