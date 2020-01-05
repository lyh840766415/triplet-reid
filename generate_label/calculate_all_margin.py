#-*-coding:GBK -*-
'''
this script is used to calculate te distance between two random node in kitti dataset
input:
	sequence of the position data
	easting, northing, altitude, orientation

algorithm step
	construct the distance graph
	compute the distance,displacement,orientation different
'''
import pygraph
from sklearn.neighbors import KDTree
import numpy as np 
import math
import pygraph.classes.graph
import pygraph.algorithms
from pygraph.algorithms.minmax import shortest_path_bellman_ford
from pygraph.algorithms.minmax import shortest_path

import pygraph.readwrite.markup as graph_write
import matplotlib.pyplot as plt
import datetime
import os

def euclidean_dis(p1,p2):
	dif_x = p1[0]-p2[0]
	dif_y = p1[1]-p2[1]
	dif_z = p1[2]-p2[2]
	return math.sqrt(dif_x*dif_x+dif_y*dif_y+dif_z*dif_z)

FILE_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/data/2011_10_03_drive_0027_sync/2011_10_03_drive_0027_sync.csv"
BASE_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/data/"

'''
function to construct Graph
	input: filename
	output 1、Graph，2、node_list
'''
def construct_Graph(filename):
	#初始化图数据结构
	G = pygraph.classes.graph.graph()
	anchor_list = np.empty(shape=[0, 4], dtype = float)
	node_list = np.empty(shape=[0, 4], dtype = float)
	last_anchor = np.empty(shape=[1,4],dtype = float)
	last_node = np.empty(shape=[1,4],dtype = float)
	node_anchor_map = np.empty(shape=[0,4],dtype = int)
	last_anchor_ind = 0

	cnt = -1

	#for verify
	#_max = 0
	#_min = 1000

	with open(filename) as file:
		while True:
			line = file.readline()
			if not line:
				cnt+=1
				break
			cnt+=1;
			G.add_node(cnt)
			key = line.split(",")
			#print("east = %.3f, north = %.3f, altitude = %.3f, yaw = %.6f"%(float(key[0]),float(key[1]),float(key[2]),float(key[3])))
			#print("cnt = %d"%(cnt))
			node_list = np.append(node_list,[[float(key[0]),float(key[1]),float(key[2]),float(key[3])]])
			
			if cnt== 0:
				anchor_list = np.append(anchor_list,[[float(key[0]),float(key[1]),float(key[2]),float(key[3])]])
				last_anchor = [float(key[0]),float(key[1]),float(key[2]),float(key[3])]
				last_node = [float(key[0]),float(key[1]),float(key[2]),float(key[3])]
				last_anchor_ind = cnt
				node_anchor_map = np.append(node_anchor_map,[[last_anchor_ind,cnt]])
				continue
			
			#判断是否为anchor点
			if(euclidean_dis(last_anchor[0:3],[float(key[0]),float(key[1]),float(key[2])])>20):
				#判断为anchor点
				#构造连接上一节点的边
				node_dis = euclidean_dis(last_node[0:3],[float(key[0]),float(key[1]),float(key[2])])
				G.add_edge((cnt-1,cnt) , wt = node_dis)
				'''
				if node_dis>_max:
					_max = node_dis
				if node_dis<_min:
					_min = node_dis
				'''
				last_node = [float(key[0]),float(key[1]),float(key[2]),float(key[3])]
				
				#构造连接上一anchor点的边				
				if not G.has_edge((last_anchor_ind,cnt)):
					anchor_dis = euclidean_dis(last_anchor[0:3],[float(key[0]),float(key[1]),float(key[2])])
					G.add_edge([last_anchor_ind,cnt],anchor_dis);
					'''
					if anchor_dis>_max:
						_max = anchor_dis
					if anchor_dis<_min:
						_min = anchor_dis
					'''
				last_anchor_ind = cnt;
				last_anchor = [float(key[0]),float(key[1]),float(key[2]),float(key[3])]
				
				#构造KDTree
				cur_anchor_list = anchor_list.reshape(int(anchor_list.shape[0]/4),4)[:,0:3];
				if not cur_anchor_list.shape[0]<2:
					tree = KDTree(anchor_list.reshape(int(anchor_list.shape[0]/4),4)[:,0:3])
					#检索KDTree 返回近距非连续anchor点
					ind_nn = tree.query_radius([[float(key[0]),float(key[1]),float(key[2])],[float(key[0]),float(key[1]),float(key[2])]],r=20)
					#判断是否为非连续点
					for ind in ind_nn[0]:
						if(anchor_list.reshape(int(anchor_list.shape[0]/4),4).shape[0]-ind>10):
							#非连续，构建连接边
							far_dis = euclidean_dis(cur_anchor_list[ind,0:3],[float(key[0]),float(key[1]),float(key[2])])
							if not G.has_edge((node_anchor_map.reshape(int(node_anchor_map.shape[0]/2),2)[ind,1],cnt)):
								'''
								if far_dis>_max:
									_max = far_dis
								if far_dis<_min:
									_min = far_dis
								'''
								G.add_edge([node_anchor_map.reshape(int(node_anchor_map.shape[0]/2),2)[ind,1],cnt] , wt = far_dis)
								if cnt - node_anchor_map.reshape(int(node_anchor_map.shape[0]/2),2)[ind,1] <= 100 :
									print("error from %d -> %d "%(cnt,node_anchor_map.reshape(int(node_anchor_map.shape[0]/2),2)[ind,1]))
									print("cur anchor = %d -> %d"%(anchor_list.shape[0],ind))
																	
				
				#将当前点加入KDTree
				anchor_list = np.append(anchor_list,[[float(key[0]),float(key[1]),float(key[2]),float(key[3])]])
				node_anchor_map = np.append(node_anchor_map,[[anchor_list.reshape(int(anchor_list.shape[0]/4),4).shape[0],cnt]])
					
			else:
				#判断不是anchor点
				node_dis = euclidean_dis(last_node[0:3],[float(key[0]),float(key[1]),float(key[2])])
				G.add_edge([cnt-1,cnt] , wt = node_dis)
				'''
				if node_dis>_max:
					_max = node_dis
				if node_dis<_min:
					_min = node_dis
				'''
				last_node = [float(key[0]),float(key[1]),float(key[2]),float(key[3])]

	return G,node_list
			
				
def save_Graph(G,node_list,filename):	
	node_list_2D = node_list.reshape(int((node_list.shape[0]/4)),4)[:,0:2]
	plt.scatter(node_list_2D[:,0],node_list_2D[:,1],s=0.2,c='green')
	print("node number = %d, edge number = %d"%(int((node_list.shape[0]/4)), len(G.edges())))

	for edge_from,edge_to in G.edges():
		plt.plot([node_list_2D[edge_from,0],node_list_2D[edge_to,0]],[node_list_2D[edge_from,1],node_list_2D[edge_to,1]],linewidth=0.5)
		
	
	plt.savefig(fname=filename,format="svg")
	plt.close("all")
	
	

def save_distmat(G,filename):
	node_num = len(G.nodes())
	distmat = np.empty(shape=[node_num, node_num], dtype = float)

	starttime = datetime.datetime.now()
	#print("-----------------------------shortest path---------------------------------")
	for i in range(node_num):
		_,dist = shortest_path(G,i)
		for j in range(node_num):
			distmat[i,j] = dist[j]
	endtime = datetime.datetime.now()
	np.savetxt(filename,distmat)

def get_distance(G):
	node_num = len(G.nodes())
	distmat = np.empty(shape=[node_num, node_num], dtype = float)

	starttime = datetime.datetime.now()
	#print("-----------------------------shortest path---------------------------------")
	for i in range(node_num):
		_,dist = shortest_path(G,i)
		for j in range(node_num):
			distmat[i,j] = dist[j]
	endtime = datetime.datetime.now()
	return distmat
	
def get_displace(nodelist):
	node_num = int((node_list.shape[0]/4))
	dispmat = np.empty(shape=[node_num, node_num], dtype = float)
	node_list_2D = node_list.reshape(int((node_list.shape[0]/4)),4)[:,0:3]
		
	for i in range(node_num):
		for j in range(node_num):
			dispmat[i,j] = euclidean_dis(node_list_2D[i,0:3],node_list_2D[j,0:3])
	
	return dispmat	

'''
获取两帧之间的角度差距
'''
def get_cori(nodelist):
	node_num = int((node_list.shape[0]/4))
	corimat = np.empty(shape=[node_num, node_num], dtype = float)
	node_list_2D = node_list.reshape(int((node_list.shape[0]/4)),4)[:,3]
	
	for i in range(node_num):
		for j in range(node_num):
			#corimat[i,j] = abs(node_list_2D[i]-node_list_2D[j])
			corimat[i,j] = math.pi-abs(abs(node_list_2D[i]-node_list_2D[j])-math.pi)
			
	
	return corimat
	
	
'''
this function is used to compute the margin that send to the triplet loss
input:
	distance_matrix
	displace_matrix
	orientetion_matrix
	
output:
	margin_matrix
'''	
def compute_margin(distmat,dispmat,corimat):
	'''
	node_num = distmat.shape[0]
	marginmat = np.empty(shape=[node_num, node_num], dtype = float)
	#for debug: weighting the disp/dist/ori
	dist_margin = np.empty(shape=[0, 2], dtype = float)
	disp_margin = np.empty(shape=[0, 2], dtype = float)
	cori_margin = np.empty(shape=[0, 2], dtype = float)	
	'''
	
	visibility = abs(distmat-dispmat)/(dispmat+0.001)
	'''
	max_visibility = np.max(visibility)
	max_dist = np.max(distmat)
	max_cori = np.max(corimat)
	'''
	max_visibility = 804.137339
	max_dist = 989.181449
	max_cori = 3.1415926
	marginmat = 1.4693*distmat/max_dist+1040.8*visibility/max_visibility+1.01*corimat/max_cori
	
	print("max_visibility = %f, max_dist = %f, mat_cori = %f"%(max_visibility,max_dist,max_cori))
	
	
	'''
	for i in range(node_num):
		for j in range(node_num):
			a=0.01
			b=0.01
			c=0.1
			index = a*dispmat[i,j]+b*abs(distmat[i,j]-dispmat[i,j])+c*corimat[i,j]
			
			
			margin = math.exp(index)
			marginmat[i,j] = margin
	'''
	
	return marginmat
	
	
'''
#Graph,nodelist = construct_Graph(FILE_PATH)

#save_Graph(Graph,nodelist)
Graph,node_list = construct_Graph(FILE_PATH)

distmat = get_distance(Graph)

dispmat = get_displace(node_list)

corimat = get_cori(node_list)

marginmat = compute_margin(distmat,dispmat,corimat)


print("-----------------------------------------------distmat--------------------------------------------------")
print(distmat)
print("-----------------------------------------------dispmat--------------------------------------------------")
print(dispmat)
print("-----------------------------------------------corimat--------------------------------------------------")
print(corimat)
print("-----------------------------------------------marginmat--------------------------------------------------")
print(marginmat)
print("-----------------------------------------------dist-disp--------------------------------------------------")
np.savetxt("DISTANCE_MAT",distmat)
np.savetxt("DISPLACE_MAT",dispmat)
np.savetxt("CORI_MAT",corimat)
np.savetxt("MARGIN_MAT",marginmat)



dif_dispdist = abs(distmat-dispmat)
print(dif_dispdist)
plt.hist(dif_dispdist)
plt.show()
plt.close("all")
plt.hist(dispmat)
plt.show()
plt.close("all")
#plt.hist(corimat)
#plt.show()
#plt.close("all")
plt.hist(marginmat)
plt.show()
plt.close("all")
'''









dirs = os.listdir(BASE_PATH)
for cur_dir in dirs :
	if os.path.isdir(os.path.join(BASE_PATH,cur_dir)):
		print(os.path.join(BASE_PATH,cur_dir))
		filename = os.path.join(BASE_PATH,cur_dir,"%s.csv"%(cur_dir))
		Graph,node_list = construct_Graph(filename)
		distmat = get_distance(Graph)
		dispmat = get_displace(node_list)
		corimat = get_cori(node_list)
		marginmat = compute_margin(distmat,dispmat,corimat)
		
		saved_MARGIN_name = os.path.join(BASE_PATH,cur_dir,"margin%s.txt"%(cur_dir))

		np.savetxt(saved_MARGIN_name,marginmat)		
		print(filename)
		print(saved_MARGIN_name)
		


