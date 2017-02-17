from keras import backend as K
import h5py
import numpy as np
import random
import os.path


#row=256
#col=256
#depth=7
class EM_Data:
	def __init__(self, patch_shape):
		assert(len(patch_shape)==3)
		self.patch_shape =patch_shape
		self.valid_data  =None
		#self.z_slices =5
	def train_generator_5D(self):
		data_dir='/tempspace/tzeng/snmes3d/data/'
		data_file='snemi3d_train_full_stacks_v1.h5'
		data_files=('snemi3d_train_full_stacks_v1.h5','snemi3d_train_full_stacks_v2.h5',
					'snemi3d_train_full_stacks_v3.h5','snemi3d_train_full_stacks_v4.h5',
					'snemi3d_train_full_stacks_v5.h5','snemi3d_train_full_stacks_v6.h5',
					'snemi3d_train_full_stacks_v7.h5','snemi3d_train_full_stacks_v8.h5',
					'snemi3d_train_full_stacks_v9.h5','snemi3d_train_full_stacks_v10.h5',
					'snemi3d_train_full_stacks_v11.h5','snemi3d_train_full_stacks_v12.h5',
					'snemi3d_train_full_stacks_v13.h5','snemi3d_train_full_stacks_v14.h5',
					'snemi3d_train_full_stacks_v15.h5','snemi3d_train_full_stacks_v16.h5')
		d_files = []
		h5fs    = []
		self.data_all =[]
		self.label_all=[]
		for i in range(len(data_files)):
			d_files.append(data_dir + data_files[i])
		for i in range(len(d_files)):
			h5fs.append(h5py.File(d_files[i],'r'))
			self.data_all.append(h5fs[i]['data'])
			self.label_all.append(h5fs[i]['label'])

		d_file = data_dir + data_file
		h5f =h5py.File(d_file,'r')
		# print('loading data file from: {0}'.format(d_file) )
		self.data=h5f['data']
		self.label=h5f['label']

		x_patch=self.patch_shape[0]
		y_patch=self.patch_shape[1]
		z_patch=self.patch_shape[2]

		x_input=self.data.shape[0]
		y_input=self.data.shape[1]
		z_input=self.data.shape[2]

		#print(x_input)
		#print(x_patch)

		#X_sec=np.zeros([1,z_patch,1,x_patch,y_patch]).astype(np.float32)
		#Y_label_outShape=[1,z_patch,2,x_patch,y_patch] #--> sample,time, channel,width, height
		#Y_sec=np.zeros(Y_label_outShape).astype(np.uint8)

		print("size of zpatch{0}".format(z_patch))
		n_files = len(self.data_all)
		start_slice =20
		while True:
			#print(x_input-x_patch)
			x_start=random.randrange(0,x_input-x_patch)
			x_end =x_start+x_patch
			y_start=random.randrange(0,y_input-y_patch)
			y_end =y_start+y_patch
			z_start=random.randrange(start_slice,z_input-z_patch)
			z_end =z_start+z_patch
			z_mid = z_start +z_patch/2
			file_id =random.randrange(0,n_files)
			#print x_start, x_end, y_start,y_end, z_start, z_end

			X=self.data_all[file_id][x_start:x_end,y_start:y_end,z_start:z_end]
			Y=self.label_all[file_id][x_start:x_end,y_start:y_end,z_start:z_end]
			X=np.transpose(X,(2,0,1))
			Y=np.transpose(Y,(2,0,1))
			X_sec =X.reshape(1,z_patch,1,x_patch,y_patch)
			Y_0 =Y.reshape(1,z_patch,1,x_patch,y_patch)
			Y_1 =1-Y_0#Y.reshape(1,z_patch,1,x_patch,y_patch)
			Y_sec=np.concatenate((Y_0,Y_1),axis=2)
			yield X_sec,Y_sec
	def valid_generator_5D(self):
		data_dir='/tempspace/tzeng/snmes3d/data/'
		data_files=('snemi3d_train_full_stacks_v1.h5','snemi3d_train_full_stacks_v2.h5')
		d_files = []
		h5fs    = []
		self.valid_data_all =[]
		self.valid_label_all=[]
		for i in range(len(data_files)):
			d_files.append(data_dir + data_files[i])
		for i in range(len(d_files)):
			h5fs.append(h5py.File(d_files[i],'r'))
			self.valid_data_all.append(h5fs[i]['data'])
			self.valid_label_all.append(h5fs[i]['label'])
		
		x_patch=self.patch_shape[0]
		y_patch=self.patch_shape[1]
		z_patch=self.patch_shape[2]

		x_input=self.data.shape[0]
		y_input=self.data.shape[1]
		z_input=self.data.shape[2]

		#X_sec=np.zeros([1,self.z_patch,1,x_patch,y_patch]).astype(np.float32)
		#Y_sec=np.zeros([1,self.z_patch,1,x_patch,y_patch]).astype(np.uint8)

		#n_files = len(self.data_all)
		n_files =len(self.valid_data_all)
		slice_end =20
		while True:
			x_start=random.randrange(0,x_input-x_patch)
			x_end =x_start+x_patch
			y_start=random.randrange(0,y_input-y_patch)
			y_end =y_start+y_patch
			z_start=random.randrange(0,slice_end-z_patch)
			z_end =z_start+z_patch
			z_mid = z_start +z_patch/2
			file_id =random.randrange(0,n_files)

			X=self.valid_data_all[file_id][x_start:x_end,y_start:y_end,z_start:z_end]
			Y=self.valid_label_all[file_id][x_start:x_end,y_start:y_end,z_start:z_end]
			X=np.transpose(X,(2,0,1))
			Y=np.transpose(Y,(2,0,1))
			X_sec =X.reshape(1,z_patch,1,x_patch,y_patch)
			
			Y_0 =Y.reshape(1,z_patch,1,x_patch,y_patch)
			Y_1 =1-Y_0#Y.reshape(1,z_patch,1,x_patch,y_patch)
			#-- notice that label is converted to 2 channel representing  
			# binary catogorical labels  so the softmax function /layer 
			# can use it to element-wisely compute SUM(y*log(y_hat))  
			# loss across pixels in the predicted probability map. 
			#  Similarly used in train_generator_5D
			Y_sec=np.concatenate((Y_0,Y_1),axis=2)
			yield X_sec,Y_sec
	def load_test_data_5D(self):
	     data_dir='/tempspace/tzeng/snmes3d/data/'
	     data_file='snemi3d_test_v1.h5'
	     #data_file='snemi3d_test_v1.h5'
	     if self.valid_data is None:
	     	d_file = data_dir + data_file
	     	h5f =h5py.File(d_file,'r')
	     	self.valid_data=h5f['data']
	     row=1024
	     col=1024
	     depth=9
	     x_size=row
	     y_size=col
	     z_size=depth
	     slices =depth
	     x_start =0
	     y_start =0
	     z_start =26

	     x_end=x_start +x_size
	     y_end=y_start +y_size
	     z_end=z_start+depth
	     X_test=np.ndarray((1, slices,1, x_size, y_size), dtype=np.float32)
	     X_test=self.valid_data[x_start:x_end,y_start:y_end,z_start:z_end]#.reshape(1,1,x_size, y_size)
	     X_test=X_test.reshape(1,1,x_size, y_size,depth)
	     X_test=np.transpose(X_test,(0,4,1,2,3))

	     return X_test


	def load_valid_data_5D(self,start_slice=0,slices=100):
	     data_dir='/tempspace/tzeng/snmes3d/data/'
	     data_file='snemi3d_train_full_stacks_v1.h5'
	     #data_file='snemi3d_test_v1.h5'
	     d_file = data_dir + data_file
	     h5f =h5py.File(d_file,'r')
	     data=h5f['data']
	     label=h5f['label']
	     row=1024
	     col=1024
	     depth=slices
	     x_size=row
	     y_size=col
	     z_size=depth
	     slices =depth
	     x_start =0
	     y_start =0
	     z_start =start_slice

	     x_end=x_start +x_size
	     y_end=y_start +y_size
	     z_end=z_start+depth
	     X_test=np.ndarray((1, slices,1, x_size, y_size), dtype=np.float32)
	     X_test=data[x_start:x_end,y_start:y_end,z_start:z_end]#.reshape(1,1,x_size, y_size)
	     X_test=X_test.reshape(1,1,x_size, y_size,depth)
	     X_test=np.transpose(X_test,(0,4,1,2,3))

	     Y_test=np.ndarray((1, slices,1, x_size, y_size), dtype=np.float32)
	     Y_test=label[x_start:x_end,y_start:y_end,z_start:z_end]#.reshape(1,1,x_size, y_size)
	     Y_test=Y_test.reshape(1,1,x_size, y_size,depth)
	     Y_test=np.transpose(Y_test,(0,4,1,2,3))

	     # time input shape is  :   num,time,chananel, height, width.

	     #  time  = deepth (sclices)

	     return X_test,Y_test