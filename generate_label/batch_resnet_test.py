import numpy as np
import os
from sklearn.neighbors import KDTree
import tensorflow as tf
import nets.resnet_v1_50 as resnet

def load_sequence(TEST_PATH,DATA_PATH,batch_size):
	seq_num = 0
	dirs = os.listdir(TEST_PATH)	
	seq_dirs=np.empty(0,dtype=object)
	for cur_dir in dirs :
		if os.path.isdir(os.path.join(TEST_PATH,cur_dir)):
			seq_num+=1	
			seq_dirs = np.append(seq_dirs,cur_dir)
	
	seq_image_size = np.empty(seq_num,dtype=np.int)
	acc_seq_image_size = np.empty(seq_num,dtype=np.int)
	for i,seq_name in enumerate(seq_dirs):
		file_path = os.path.join(DATA_PATH,seq_name,seq_name[0:10],seq_name,"image_02/data")
		seq_image_size[i] = len(os.listdir(file_path))
		if i == 0:
			acc_seq_image_size[0] = seq_image_size[0]
		else:
			acc_seq_image_size[i] = acc_seq_image_size[i-1]+seq_image_size[i]
	tot_image_num = np.sum(seq_image_size)
	
	tot_image_num_batch = (1+tot_image_num//batch_size[0])*batch_size[0]
	image_fullpath=np.empty(tot_image_num_batch, dtype=object)
	cnt = 0
	for i,seq_name in enumerate(seq_dirs):
		file_path = os.path.join(DATA_PATH,seq_name,seq_name[0:10],seq_name,"image_02/data")
		images_name = os.listdir(file_path)
		for name in images_name:
			image_fullpath[cnt] = os.path.join(DATA_PATH,seq_name,seq_name[0:10],seq_name,"image_02/data",name)
			cnt+=1
	
	for i in range(tot_image_num_batch-tot_image_num):
		image_fullpath[tot_image_num+i]=image_fullpath[tot_image_num-1]

	return seq_num,seq_dirs,tot_image_num,seq_image_size,acc_seq_image_size,image_fullpath

def load_pos(seq_name,TEST_PATH):
	filename = os.path.join(TEST_PATH,seq_name,"%s.csv"%(seq_name))
	lines = len(open(filename).readlines())
	pos_list = np.empty(shape=[lines, 3], dtype = np.float)
	
	cnt = -1
	with open(filename) as file:
		while True:
			line = file.readline()
			if not line:
				cnt+=1
				break
			cnt+=1;
			key = line.split(",")
			#print("east = %.3f, north = %.3f, altitude = %.3f, yaw = %.6f"%(float(key[0]),float(key[1]),float(key[2]),float(key[3])))
			#print("cnt = %d"%(cnt))
			pos_list[cnt,:] = [float(key[0]),float(key[1]),float(key[2])]
	return pos_list,lines
		
#Contain the query	
def load_neighbour(testset_size,acc_seq_image_size,seq_dirs,seq_image_size,TEST_PATH,neighbor_dist):
	neighbor_list = [None]*testset_size
	
	start_index = 0
	end_index = 0
	for i,seq_name in enumerate(seq_dirs):
		end_index = acc_seq_image_size[i]
		#load pos data
		pos_list,lines = load_pos(seq_name,TEST_PATH)
		if lines != seq_image_size[i]:
			print("Inconsistent number of images")
		#construct the kd-tree
		tree = KDTree(pos_list)
		
		neighbor_ind,_ = tree.query_radius(pos_list, neighbor_dist,sort_results=True,return_distance=True)
		neighbor_list[start_index:] = neighbor_ind + start_index
		#print(neighbor_list[end_index-1])
		#print(seq_name)
		
		#final update the start index
		start_index = acc_seq_image_size[i]
	return neighbor_list

def match_fid_label(label,fids):
	return label[1],fids[label[0]]
	
def fid_to_image(fid,image_size):
	image_encoded = tf.read_file(fid)
	image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
	image_resized = tf.image.resize_images(image_decoded, image_size)
	return image_resized,fid
	
def init_dataset_loader(path,batch_size):
	loading_threads = 10
	net_input_size = (144, 288)
	dataset = tf.data.Dataset.from_tensor_slices(path)
	dataset = dataset.map(lambda fid: fid_to_image(fid,image_size=net_input_size),num_parallel_calls=loading_threads)
	dataset = dataset.batch(batch_size[0])
	iter = dataset.make_initializable_iterator()
	images,fids = iter.get_next()
	return images,fids,iter


#TO DO: load model parameter
def init_network(images,embbed_size):
	endpoints,body_prefix=resnet.endpoints(images,is_training=True)
	#fix output feature to 128-d
	feature = tf.layers.dense(endpoints['model_output'], embbed_size)
	return feature

def load_feature(image_filename,embbed_size,LOG_PATH,batch_size,testset_size,MODEL_PATH,MODEL_NAME):
	#init dataset loader
	filepath_placeholder = tf.placeholder(tf.string,shape=[None])
	batch_per_run_placeholder = tf.placeholder(tf.int64,shape=[1])
	images, fids, iter = init_dataset_loader(filepath_placeholder,batch_per_run_placeholder)
	
	
	#init network
	feature = init_network(images,embbed_size)
	
	
	#compute feature
	#config GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	
	#restore parameter from saved model
	model_loader = tf.train.Saver()
	all_feature = np.empty([0,embbed_size],np.float32)
	#Start training
	with tf.Session(config=config) as sess:
		#load the model patameter
		model_loader = tf.train.import_meta_graph(os.path.join(MODEL_PATH,MODEL_NAME))
		model_loader.restore(sess,"/home/luyuheng/lab/triplet-reid/generate_label/Model/batch_resnet_triplet_accuracy_iter_009000_model.ckpt")
		
		summary_writer = tf.summary.FileWriter(LOG_PATH,sess.graph)
		
		train_feed_dict = {
			filepath_placeholder: image_filename,
			batch_per_run_placeholder:batch_size
		}
		
		sess.run(iter.initializer,feed_dict=train_feed_dict)
		
		iter_num = (testset_size//batch_size[0])+1
		for step in range(iter_num):
			run_feature = sess.run([feature])
			all_feature = np.concatenate((all_feature,run_feature[0]))
			#print(run_feature)
			
			print(all_feature.shape)
			
			
	return all_feature	

def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]
	
def compute_topk_dist(feature_list,testset_size,top_k):
	dist_list = [None]*testset_size
	#distance between feature
	for i in range(testset_size):
		dist = np.linalg.norm(feature_list[i,:]-feature_list[0:testset_size,:],ord=2,axis=1)
		dist = dist.reshape(1,dist.shape[0])
		dist_topk = partition_arg_topK(dist,top_k+1,1)
		dist_list[i] = dist_topk
		#print(i,dist_list[i])
		#print(dist_list[i].shape)
	return dist_list	
		
def compute_acc_recall(feature_list,label_list,testset_size,top_k):
	dist_list = compute_topk_dist(feature_list,testset_size,top_k)
	tot_recall = 0
	tot_accuracy = 0
	#accuracy,recall = compute_acc_recall()
	for i in range(testset_size):
		intersect = np.intersect1d(dist_list[i], label_list[i], assume_unique=True)
		intersect1 = np.intersect1d(intersect,i)
		#print(intersect)
		#print(intersect1)
		setdiff = np.setdiff1d(intersect,i)
		recall = setdiff.shape[0]/(label_list[i].shape[0]-1)
		tot_recall += recall
		accuracy = setdiff.shape[0]/top_k
		tot_accuracy += accuracy
		#print("accuracy = %f,recall = %f"%(accuracy,recall))
	
	tot_accuracy = tot_accuracy/testset_size
	tot_recall = tot_recall/testset_size
	#print("accuracy_top_%d = %f,recall_top_%d = %f"%(top_k,tot_accuracy/testset_size,top_k,tot_recall/testset_size))
	return tot_accuracy,tot_recall
	

def main():
	#define the test set path
	neighbor_dist = 20
	embbed_size = 128	
	batch_size = np.empty(1,dtype = np.int)
	batch_size[0] = 10
	top_k = 1
	DATA_PATH = "/home/luyuheng/lab/dataset/KITTI_RAW"
	TEST_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/data/"
	LOG_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/experiment2"
	MODEL_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/Model/"
	MODEL_NAME = "batch_resnet_triplet_accuracy_iter_009000_model.ckpt.meta"

	#test set contains of a total of X images,each image has a index.
	seq_num,seq_dirs,testset_size,seq_image_size,acc_seq_image_size,image_fullpath = load_sequence(TEST_PATH,DATA_PATH,batch_size)
	print(testset_size)

	#load the neighbour,contain the query
	neighbor_list = load_neighbour(testset_size,acc_seq_image_size,seq_dirs,seq_image_size,TEST_PATH,neighbor_dist)
	print(neighbor_list[10].shape[0])

	feature_list = load_feature(image_fullpath,embbed_size,LOG_PATH,batch_size,testset_size,MODEL_PATH,MODEL_NAME)
	print(feature_list.shape)
	print("Pause")
	#input()
	#feature list size error
	
	#dist_list = compute_topk_dist(feature_list,testset_size,top_k)	
	
	top_acc_recall = np.empty([1000,3],np.float32)
	#accuracy,recall = compute_acc_recall(feature_list,neighbor_list,testset_size,top_k)
	for i in range(1,1000):
		accuracy,recall = compute_acc_recall(feature_list,neighbor_list,testset_size,i)
		top_acc_recall[i,0] = i
		top_acc_recall[i,1] = accuracy
		top_acc_recall[i,2] = recall
		print("top_%d accuracy = %f recall = %f"%(i,accuracy,recall))
		if i%10 == 0:
			np.savetxt("top_acc_recall_%d.txt"%(i), top_acc_recall)
		
	
if __name__ == '__main__':
	main()
