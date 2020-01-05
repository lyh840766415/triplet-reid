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

def euclidean_dis(p1,p2):
	dif_x = p1[0]-p2[0]
	dif_y = p1[1]-p2[1]
	dif_z = p1[2]-p2[2]
	return math.sqrt(dif_x*dif_x+dif_y*dif_y+dif_z*dif_z)

FILE_PATH = "/home/luyuheng/lab/triplet-reid/generate_label/data/2011_09_26_drive_0029_sync/2011_09_26_drive_0029_sync.csv"

#初始化图数据结构
G = pygraph.classes.graph.graph()

anchor_list = np.empty(shape=[0, 4], dtype = float)
node_list = np.empty(shape=[0, 4], dtype = float)
last_anchor = np.empty(shape=[1,4],dtype = float)
last_node = np.empty(shape=[1,4],dtype = float)
node_anchor_map = np.empty(shape=[0,4],dtype = int)
last_anchor_ind = 0

cnt = -1

_max = 0
_min = 1000

with open(FILE_PATH) as file:
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
			#print(euclidean_dis(last_anchor[0:3],[float(key[0]),float(key[1]),float(key[2])]))
			#构造连接上一节点的边
			node_dis = euclidean_dis(last_node[0:3],[float(key[0]),float(key[1]),float(key[2])])
			G.add_edge((cnt-1,cnt) , wt = node_dis)
			#print("wt = %f"%(node_dis))
			'''
			if node_dis>_max:
				_max = node_dis
			if node_dis<_min:
				_min = node_dis
			'''
			#print("case 1 add edge")
			last_node = [float(key[0]),float(key[1]),float(key[2]),float(key[3])]
			
			#构造连接上一anchor点的边
			#print(not G.has_edge((last_anchor_ind,cnt)))
			
			if not G.has_edge((last_anchor_ind,cnt)):
				anchor_dis = euclidean_dis(last_anchor[0:3],[float(key[0]),float(key[1]),float(key[2])])
				G.add_edge([last_anchor_ind,cnt],anchor_dis);
				'''
				if anchor_dis>_max:
					_max = anchor_dis
				if anchor_dis<_min:
					_min = anchor_dis
				'''
				#print("case 2 add edge")
			last_anchor_ind = cnt;
			last_anchor = [float(key[0]),float(key[1]),float(key[2]),float(key[3])]
			
			#构造KDTree
			#print(anchor_list.shape)
			#if anchor_list.shape[0]>1 :
			#	print("shape > 1")
			cur_anchor_list = anchor_list.reshape(int(anchor_list.shape[0]/4),4)[:,0:3];
			#print(cur_anchor_list.shape)
			#print(cur_anchor_list)
			if not cur_anchor_list.shape[0]<2:
				#print("debug place 1")
				tree = KDTree(anchor_list.reshape(int(anchor_list.shape[0]/4),4)[:,0:3])
				#检索KDTree 返回近距非连续anchor点
				ind_nn = tree.query_radius([[float(key[0]),float(key[1]),float(key[2])],[float(key[0]),float(key[1]),float(key[2])]],r=20)
				#print(ind_nn[0])
				#判断是否为非连续点
				for ind in ind_nn[0]:
					#print("ind = %d"%(ind))
					if(anchor_list.reshape(int(anchor_list.shape[0]/4),4).shape[0]-ind>10):
						#print("anchor_list = %d, ind = %d dif = %d"%(anchor_list.shape[0],ind,(anchor_list.shape[0]-ind)))
						#非连续，构建连接边
						#print(cur_anchor_list)
						far_dis = euclidean_dis(cur_anchor_list[ind,0:3],[float(key[0]),float(key[1]),float(key[2])])
						#print(node_anchor_map)
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
							#print("case 3 add edge")		
								
			
			#将当前点加入KDTree
			anchor_list = np.append(anchor_list,[[float(key[0]),float(key[1]),float(key[2]),float(key[3])]])
			node_anchor_map = np.append(node_anchor_map,[[anchor_list.reshape(int(anchor_list.shape[0]/4),4).shape[0],cnt]])
				
		else:
			#判断不是anchor点
			node_dis = euclidean_dis(last_node[0:3],[float(key[0]),float(key[1]),float(key[2])])
			G.add_edge([cnt-1,cnt] , wt = node_dis)
			#print("wt = %f"%(node_dis))
			'''
			if node_dis>_max:
				_max = node_dis
			if node_dis<_min:
				_min = node_dis
			'''
			#print("case 4 add edge")
			last_node = [float(key[0]),float(key[1]),float(key[2]),float(key[3])]
			
				
def draw_Graph(G,node_list,anchor_list):
	
	
	node_list_2D = node_list.reshape(int((node_list.shape[0]/4)),4)[:,0:2]
	plt.scatter(node_list_2D[:,0],node_list_2D[:,1],s=0.2,c='green')
	
	print("node number = %d, edge number = %d"%(int((node_list.shape[0]/4)), len(G.edges())))
	#print(G.edges())

	for edge_from,edge_to in G.edges():
		plt.plot([node_list_2D[edge_from,0],node_list_2D[edge_to,0]],[node_list_2D[edge_from,1],node_list_2D[edge_to,1]],linewidth=0.5)
		
	
	plt.savefig(fname="save.svg",format="svg")
		
	#plt.show()
	

draw_Graph(G,node_list,anchor_list)


#calculate distance
distmat = np.empty(shape=[cnt, cnt], dtype = float)

'''
starttime = datetime.datetime.now()
print("-----------------------------shortest path---------------------------------")
for i in range(cnt):
	_,dist = shortest_path_bellman_ford(G,i)
	for j in range(cnt):
		distmat[i,j] = dist[j]
endtime = datetime.datetime.now()
print("bellman_ford = %d "%((endtime - starttime).seconds))
'''

starttime = datetime.datetime.now()
print("-----------------------------shortest path---------------------------------")
for i in range(cnt):
	_,dist = shortest_path(G,i)
	for j in range(cnt):
		distmat[i,j] = dist[j]
endtime = datetime.datetime.now()
print("Dijkstra = %d "%((endtime - starttime).seconds))

print(distmat)

np.savetxt("dist.txt",distmat)

		



#print(node_anchor_map.reshape(int(node_anchor_map.shape[0]/2),2))

#print("_max = %f, _min = %f"%(_max,_min))


#初始化图数据结构

#按行读入序列文件

#判断是否为anchor点

	#若是anchor点
		#构建一条边连接上一个节点与当前节点，权重为两点间距离
		#构建一条边连接上一个anchor，权重为两anchor间距离。
		#搜索kd-tree中特定范围中，所有10米内临近anchor点，要求两点间路径大于20米
		#连接当前anchor与所有满足要求的anchor点。
		#将当前anchor点加入kd-tree
	
	#若不是anchor点
		#构建一条边连接上一个节点与当前节点，权重为两点间距离

