from keras.utils import np_utils, visualize_util
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from theano import tensor as T
from keras.optimizers import RMSprop, SGD
from keras.callbacks  import CSVLogger
from keras.layers import Input,merge
from keras.models import Model
import h5py
import numpy as np
import random
import os.path
import zt_kdd_em_models as my_model
from EM_Data import EM_Data
import matplotlib.pyplot as plt
from sklearn import metrics
import math


# -- Store dict the model file name and  model creation function --
model_dict={}

# -------------------- DeepEM3D (LSTM,GRU) variants --------------------------------------------------------
model_dict['deepEM3D_LSTM']=['deepEM3D_Deconv_LSTM',my_model.timeDist_DeepEM3D_Net_DeconvLSTM]
model_dict['deepEM3D_GRU']=['deepEM3D_Deconv_GRU',my_model.timeDist_DeepEM3D_Net_DeconvGRU]

model_dict['deepEM2D']=['deepEM2D',my_model.timeDist_DeepEM2D_Net]
model_dict['deepEM3D']=['deepEM3D',my_model.timeDist_DeepEM3D_Net]
model_dict['deepEM3D_simple_UP']=['deepEM3D_simpleUP',my_model.timeDist_DeepEM3D_Net_simpleUP]
model_dict['test_deconv']=['test_deconv',my_model.model_test_deconv]

# ---------------------  Unet (LSTM,GRU) vatiant---------------------------------------------#
model_dict['time_unet_LSTM_1_level']=['unet2D_LSTM_1_level',my_model.time_LSTM_unet_1_level]
model_dict['time_unet_LSTM_3_level']=['unet2D_LSTM_3_level',my_model.time_LSTM_unet_3_level]
model_dict['time_unet_LSTM_5_level']=['unet2D_LSTM_5_level',my_model.time_LSTM_unet_5_level]


# --------------------------------------------Unet GRU---------------------------------------------#
model_dict['time_unet2D']=['unet2D',my_model.time_unet2D]

model_dict['time_unet_GRU_1_level']=['unet2D_RGU_1_level',my_model.time_GRU_unet_1_level]
model_dict['time_unet_GRU_3_level']=['unet2D_RGU_3_level',my_model.time_GRU_unet_3_level]
model_dict['time_unet_GRU_5_level']=['unet2D_RGU_5_level',my_model.time_GRU_unet_5_level]

#--define  patch size --
row=256
col=256
depth=7

def soft_max_loss(y_true,y_pred):
	
	y_cal=y_true*y_pred
	y_cal_max=K.sum(y_cal,axis=2)
	return -K.sum(K.log(K.clip(y_cal_max,K.epsilon(),1-K.epsilon())))/y_true.size#

def myacc(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	return K.mean(K.equal(y_true_f, K.round(y_pred_f)))


def train_model(model_name_key):
	if not model_dict.has_key(model_name_key):
		print "can not find model {}".format(model_name_key)
		return
	model_name       = model_dict[model_name_key][0]
	model_create_func = model_dict[model_name_key][1]
	EM_D =EM_Data(patch_shape=(row,col,depth))
	ip = Input(shape=(None,1, row, col))
	# model_dict[model][1] return a function pointer for ceating network model
	output =model_create_func(ip)
	model  =Model(ip,output)
	#weight_h5_file="./deconv_GRU_unet_test_temp.h5"
	weight_h5_file='./'+ model_name +'.h5'
	if os.path.isfile(weight_h5_file):
		try:
			model.load_weights(weight_h5_file)
		except:
			print 'the model {} can not  be loaded'.format(weight_h5_file)
			pass
	
	optimizer =RMSprop(lr = 1e-6)
	model.compile(optimizer=optimizer, loss=soft_max_loss,metrics=[myacc])
	model.summary()
	visualize_util.plot(model,model_name +'.png',show_shapes=True)
	best_model = ModelCheckpoint(weight_h5_file, verbose = 1, save_best_only = True)
	csv_logger = CSVLogger('./'+model_name +'.log')
	history=model.fit_generator(EM_D.train_generator_5D(), samples_per_epoch = 300, nb_epoch = 600, verbose=1 ,nb_worker=1, 
						nb_val_samples=60, validation_data=EM_D.valid_generator_5D(),callbacks = [best_model,csv_logger])
	return history

def predict_model(model_name_key):
	r=1024
	c=1024
	d=20
	if not model_dict.has_key(model_name_key):
		print "can not find model {}".format(model_name_key)
		return
	model_name       = model_dict[model_name_key][0]
	model_create_func = model_dict[model_name_key][1]
	EM_D =EM_Data(patch_shape=(r,c,d))
	
	ip = Input(shape=(None,1, r,c))
	output =model_create_func(ip)
	model  =Model(ip,output)


	weight_h5_file='./'+ model_name +'.h5'
	if os.path.isfile(weight_h5_file):
		try:
			model.load_weights(weight_h5_file)
		except:
			print 'the model {} can not  be loaded'.format(weight_h5_file)
			return
	iter_num = int(math.ceil(20.0/float(d)))
	for i in range(iter_num):
		X_test,Y_test=EM_D.load_valid_data_5D(start_slice=i*d,slices =d)
		print X_test.shape,Y_test.shape
		Y_predict=model.predict(X_test)
		print Y_predict.shape
		#print "predict {}".foramt(i)
		if i==0:
			Y_P=Y_predict
			Y_T=Y_test
		else:
			Y_P=np.concatenate((Y_P,Y_predict),axis=1)
			Y_T=np.concatenate((Y_T,Y_test),axis=1)
	Y_P=Y_P[:,:,0]
	#print Y_P.shape, Y_T.shape
	#A=np.sum((np.round(Y_P)-Y_T))
	#print A.shape
	P =Y_P.reshape(Y_P.size,)
	T =Y_T.reshape(Y_T.size,)
	Acurracy = np.mean(np.round(P)==T)
	fpr, tpr, thresholds = metrics.roc_curve(T, P)
	auc=metrics.auc(fpr, tpr)
		#scores = model.evaluate(X_test, Y_test, verbose=0)
	return Acurracy,auc,Y_P, Y_T
def savePredictIMG(Y,directory):
	dim =YP.ndim
	if not os.path.exists(directory):
		os.makedirs(directory)
	slices = Y.shape[1]
	if dim == 4:
		row =Y.shape[2]
		col =Y.shape[3]
	elif dim ==5:
		row =Y.shape[3]
		col =Y.shape[4]
	else:
		raise ValueError('prob map stack demension must between 4-5, you give {}'.format(dim))

	for i in range(slices):
		im=Y[0,i,0].reshape(row,col) if dim ==5 else Y[0,i].reshape(row,col)
		im=1-im
		fig=plt.imshow(im,cmap='Greys')
		plt.axis('off')
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		file_name=directory+'/prdict_{}.pdf'.format(i)
		print('write to '+ file_name)
		plt.savefig(file_name, bbox_inches='tight', pad_inches = 0)

		file_name=directory+'/prdict_{}.png'.format(i)
		print('write to '+ file_name)
		plt.savefig(file_name, bbox_inches='tight', pad_inches = 0)
			#plt.show()

def save_prob_to_hd5(Y,directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
        dim=Y.ndim
        if dim == 4:
		row =Y.shape[2]
		col =Y.shape[3]
	elif dim ==5:
		row =Y.shape[3]
		col =Y.shape[4]
	else:
		raise ValueError('prob map stack demension must between 4-5, you give {}'.format(dim))
	assert Y.shape[0]==1, "can't save prob data haveing more than 1 sample"
	depth = Y.shape[1]
    
        Y=Y.reshape(depth,row,col)
	Y=np.transpose(Y,(1,2,0))
	print Y.shape
	d_file = directory+'/prob.h5'
	h5f=h5py.File(d_file,'w')
	dset = h5f.create_dataset('data', data=Y)
	h5f.close()
def adapted_rand(gt_seg,pred_seg):
	from cremi.evaluation import rand
	return rand.adapted_rand(gt_seg,pred_seg)
def water_shad(prob_b):
	from scipy import ndimage as ndi
	from skimage.morphology import watershed
	from skimage.feature import peak_local_max
	binary_boundary = np.round(prob_b)
	distance = ndi.distance_transform_edt(binary_boundary)
	local_maxi = peak_local_max(distance, indices=False,
                            labels=binary_boundary)
	markers = ndi.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=binary_boundary)
	return labels

if __name__ == "__main__":
	models={}
	models['deepEM3D_LSTM']='deepEM3D_LSTM'
	
        # Uncomment rest below for other models
        #---------------------------------------------------------#
	#models['time_unet_GRU_3_level']='time_unet_GRU_3_level'
	#models['time_unet_GRU_5_level']='time_unet_GRU_5_level'
	#models['time_unet_LSTM_1_level']='time_unet_LSTM_1_level'
	# models['time_unet_LSTM_3_level']='time_unet_LSTM_3_level'
	# models['time_unet_LSTM_5_level']='time_unet_LSTM_5_level'
	#models['deepEM3D_GRU']='deepEM3D_GRU'
	#models['deepEM3D']='deepEM3D'
	#model_name='deepEM3D_GRU'
	#model_name='deepEM2D'
	#model_name='deepEM3D'
	#model_name='deepEM3D_simple_UP'
	#model_name='test_deconv'
	phase = 'train'
	if phase is 'train':
                # uncomment for a particular model prediction
                #model_name='deepEM3D_GRU'
                #model_name='deepEM2D'
                #model_name='deepEM3D'
                #model_name='deepEM3D_simple_UP'
                #model_name='test_deconv'
		#model_name ='deepEM2D'

        	#model_name ='time_unet2D'
        	#model_name ='time_unet_GRU_1_level'
        	#model_name ='time_unet_GRU_3_level'
        	#model_name ='time_unet_GRU_5_level'
        	#model_name ='time_unet_LSTM_3_level'
        	#model_name ='time_unet_LSTM_5_level'
        	#model_name ='time_unet_LSTM_1_level'
		history=train_model(model_name)
	elif phase is 'predict':
	     for model_name in models:
		 print model_name
		 Accuracy, AUC, YP, YT=predict_model(model_name)
		 Y_3D  =YP.reshape(20,1024,1024)
		 #Y_3D  =np.transpose(Y_3D,(1,2,0))
	         Y_T3D =YT.reshape(20,1024,1024)
	         #Y_T3D =np.transpose(Y_3D,(1,2,3))
	         print Y_T3D.shape, Y_3D.shape
	         p_r_e=adapted_rand(Y_T3D.astype(int),np.round(Y_3D).astype(int))
	         print 'ACC={} , AUC ={}, pixel_rand_err ={}'.format(Accuracy,AUC,p_r_e)
	        #pred_img_folder= model_name
	         prob_folder ="./"+ model_name +"valid_prob_img"
	         h5_folder   ="./"+model_name + "valid_prob_h5"
	         savePredictIMG(YP,prob_folder)
	         save_prob_to_hd5(YP,h5_folder)

	
	#history=train_model(model_name)
