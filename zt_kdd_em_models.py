from keras.layers import merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.convolutional import MaxPooling2D, Convolution3D, Convolution2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D, UpSampling2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Permute
from keras.layers.recurrent import LSTM,GRU
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.extra_conv_recurrent import DeconvLSTM2D, ConvGRU2D, DeconvGRU2D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.SpatialPyramidPooling import SpatialPyramidPooling

from keras import backend as K
from theano import tensor as T


K.set_image_dim_ordering("th")
if K.image_dim_ordering() == "th":
    channel_axis_4d = 1
    channel_axis_5d = 2
else:
    channel_axis_4d = -1
    channel_axis_5d = -1


# -- BottleNeck Block of convulutional RNN /(conv_LSTM,conv_GRU, Deconv_lstm, Deconv_GRU) --------------------------------------#
def time_ConvLSTM_bottleNeck_block(x,filters,row,col):
    reduced_filters =filters
    if filters >=8:
        reduced_filters =int(round(filters/8))
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu',border_mode='same'))(x)
    x = Bidirectional(ConvLSTM2D(nb_filter=reduced_filters, nb_row=row, nb_col=col, dim_ordering='th', border_mode='same',return_sequences=True),merge_mode='sum')(x)
    x = TimeDistributed(Convolution2D(nb_filter=filters, nb_row=1, nb_col=1, activation='relu',border_mode='same'))(x)	
    return x

def time_ConvGRU_bottleNeck_block(x,filters,row,col):
    reduced_filters =filters
    if filters >=8:
        reduced_filters =int(round(filters/8))
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu',border_mode='same'))(x)
    x = Bidirectional(ConvGRU2D(nb_filter=reduced_filters, nb_row=row, nb_col=col, dim_ordering='th', border_mode='same',return_sequences=True),merge_mode='sum')(x)
    x = TimeDistributed(Convolution2D(nb_filter=filters, nb_row=1, nb_col=1, activation='relu',border_mode='same'))(x)	
    return x

def time_DeconvLSTM_bottleNeck_block(x,filters,row,col,subsample=(2,2)):
    reduced_filters =filters
    if filters >=8:
        reduced_filters =int(round(filters/8))
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu',border_mode='same'))(x)
   
    x1 = DeconvLSTM2D(nb_filter=reduced_filters, nb_row=row, nb_col=col,go_backwards=True, subsample=subsample, dim_ordering='th', border_mode='same',return_sequences=True)(x)
    x2 = DeconvLSTM2D(nb_filter=reduced_filters, nb_row=row, nb_col=col,go_backwards=False, subsample=subsample, dim_ordering='th', border_mode='same',return_sequences=True)(x)
    x=merge([x1,x2],mode='sum',concat_axis=2)
    x = TimeDistributed(Convolution2D(nb_filter=filters, nb_row=1, nb_col=1, activation='relu',border_mode='same'))(x)	
    return x

def time_DeconvGRU_bottleNeck_block(x,filters,row,col,subsample=(2,2)):
    reduced_filters =filters
    if filters >=8:
        reduced_filters =int(round(filters/8))
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu',border_mode='same'))(x)
    x1 = DeconvGRU2D(nb_filter=reduced_filters, nb_row=row, nb_col=col,go_backwards=True, subsample=subsample, dim_ordering='th', border_mode='same',return_sequences=True)(x)
    x2 = DeconvGRU2D(nb_filter=reduced_filters, nb_row=row, nb_col=col,go_backwards=False, subsample=subsample, dim_ordering='th', border_mode='same',return_sequences=True)(x)
    x=merge([x1,x2],mode='sum',concat_axis=2)
    x = TimeDistributed(Convolution2D(nb_filter=filters, nb_row=1, nb_col=1, activation='relu',border_mode='same'))(x)	
    return x



# 2D soft max for 5D [#batch,#time,#channel,#height,#width] (time distributed )  data

def time_dist_softmax(x):
    assert K.ndim(x)==5
    #e = K.exp(x - K.max(x, axis=2, keepdims=True))
    e = K.exp(x)
    s = K.sum(e, axis=2, keepdims=True)
    return e / s

def time_dist_softmax_out_shape(input_shape):
    shape=list(input_shape)
    return tuple(shape)

#  Inception -- Residual modules for time distributed structures ---------#
def time_inception_resnet_stem_3Dconv(input):
    # Input Shape is 320 x 320 x 3 (tf) or 3 x 320 x 320 (th)

    ## for original our baimatica paper, we used 2 layer of 3D convolution followed by 2D conv
    # here we do 2 layer 3D conv, but keep the times (original #slices in Z direction) unchanged.

    # First, permute time 5d time to the order : [ch,height,weidth,time(Z)] for 3D covlution
    c = Permute([2,3,4,1])(input)
    c = Convolution3D(32, 3, 3, 3, activation='relu', subsample=(2,2,1),border_mode='same')(c)
	#next_size= 160#input[1]/2
    c = Convolution3D(64, 3, 3, 3, activation='relu', subsample=(1,1,1),border_mode='same')(c)
    # Now, for rest of time distributed 2D convs, we need Permute data back to dimention order [time,channel,height, width]
    c = Permute([4,1,2,3])(c)
	
    c = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(c)
    # --------------  left branch -------------------------------------------
    c1 = TimeDistributed(Convolution2D(64, 1, 1, activation='relu', border_mode='same'))(c)
    c1 = TimeDistributed(Convolution2D(96, 3, 3, activation='relu', border_mode='same'))(c1)
    # --------------- right branch ------------------------------------------
    c2 = TimeDistributed(Convolution2D(64, 1, 1, activation='relu', border_mode='same'))(c)
    c2 = TimeDistributed(Convolution2D(64, 7, 1, activation='relu', border_mode='same'))(c2)
    c2 = TimeDistributed(Convolution2D(64, 1, 7, activation='relu', border_mode='same'))(c2)
    c2 = TimeDistributed(Convolution2D(96, 3, 3, activation='relu', border_mode='same'))(c2)
    m1 = merge([c1, c2], mode='concat', concat_axis=channel_axis_5d)
	#m_pad =ZeroPadding2D((1,1))(m1)
    p1 = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2),border_mode='same'))(m1)
    p2 = TimeDistributed(Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2),border_mode='same'))(m1)
    m2 = merge([p1, p2], mode='concat', concat_axis=channel_axis_5d)
    m2 = BatchNormalization(axis=2)(m2)
    m2 = Activation('relu')(m2)
    return m2, m1


def time_inception_resnet_stem_2Dconv(input):
    # Input Shape is 320 x 320 x 3 (tf) or 3 x 320 x 320 (th)
	#c = ZeroPadding3D((1,1,0))(input)
    c = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', subsample=(2,2),border_mode='same'))(input)
    c = BatchNormalization(axis=2)(c)
	
    c = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(c)
    c = BatchNormalization(axis=2)(c)
    # --------------  left branch -------------------------------------------
    c1 = TimeDistributed(Convolution2D(64, 1, 1, activation='relu', border_mode='same'))(c)
    c1 = BatchNormalization(axis=2)(c1)
    c1 = TimeDistributed(Convolution2D(96, 3, 3, activation='relu', border_mode='same'))(c1)
    c1 = BatchNormalization(axis=2)(c1)
    # --------------- right branch ------------------------------------------
    c2 = TimeDistributed(Convolution2D(64, 1, 1, activation='relu', border_mode='same'))(c)
    c2 = BatchNormalization(axis=2)(c2)
    c2 = TimeDistributed(Convolution2D(64, 7, 1, activation='relu', border_mode='same'))(c2)
    c2 = BatchNormalization(axis=2)(c2)
    c2 = TimeDistributed(Convolution2D(64, 1, 7, activation='relu', border_mode='same'))(c2)
    c2 = BatchNormalization(axis=2)(c2)
    c2 = TimeDistributed(Convolution2D(96, 3, 3, activation='relu', border_mode='same'))(c2)
    m1 = merge([c1, c2], mode='concat', concat_axis=channel_axis_5d)
	#m_pad =ZeroPadding2D((1,1))(m1)
    p1 = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2),border_mode='same'))(m1)
    p2 = TimeDistributed(Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2),border_mode='same'))(m1)
    m2 = merge([p1, p2], mode='concat', concat_axis=channel_axis_5d)
    m2 = BatchNormalization(axis=2)(m2)
    m2 = Activation('relu')(m2)
    return m2, m1

def time_inception_resnet_v2_A(input, scale_residual=True):
    # Input is relu activation
    init = input

    ir1 = TimeDistributed(Convolution2D(32, 1, 1, activation='relu', border_mode='same'))(input)

    ir2 = TimeDistributed(Convolution2D(32, 1, 1, activation='relu', border_mode='same'))(input)
    ir2 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(ir2)

    ir3 = TimeDistributed(Convolution2D(32, 1, 1, activation='relu', border_mode='same'))(input)
    ir3 = TimeDistributed(Convolution2D(48, 3, 3, activation='relu', border_mode='same'))(ir3)
    ir3 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(ir3)

    ir_merge = merge([ir1, ir2, ir3], concat_axis=channel_axis_5d, mode='concat')

    ir_conv = TimeDistributed(Convolution2D(384, 1, 1, activation='linear', border_mode='same'))(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=2)(out)
    out = Activation("relu")(out)
    return out


def time_inception_resnet_v2_B(input, scale_residual=True):
    # Input is relu activation
    init = input

    ir1 = TimeDistributed(Convolution2D(192, 1, 1, activation='relu', border_mode='same'))(input)

    ir2 = TimeDistributed(Convolution2D(128, 1, 1, activation='relu', border_mode='same'))(input)
    ir2 = TimeDistributed(Convolution2D(160, 1, 7, activation='relu', border_mode='same'))(ir2)
    ir2 = TimeDistributed(Convolution2D(192, 7, 1, activation='relu', border_mode='same'))(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=channel_axis_5d)

    ir_conv = TimeDistributed(Convolution2D(1152, 1, 1, activation='linear', border_mode='same'))(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=2)(out)
    out = Activation("relu")(out)
    return out

def time_inception_resnet_v2_C(input, scale_residual=True):
    # Input is relu activation
    init = input

    ir1 = TimeDistributed(Convolution2D(192, 1, 1, activation='relu', border_mode='same'))(input)

    ir2 = TimeDistributed(Convolution2D(192, 1, 1, activation='relu', border_mode='same'))(input)
    ir2 = TimeDistributed(Convolution2D(224, 1, 3, activation='relu', border_mode='same'))(ir2)
    ir2 = TimeDistributed(Convolution2D(256, 3, 1, activation='relu', border_mode='same'))(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=channel_axis_5d)

    ir_conv = TimeDistributed(Convolution2D(2144, 1, 1, activation='linear', border_mode='same'))(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=2)(out)
    out = Activation("relu")(out)
    return out



def time_reduction_A(input, k=192, l=224, m=256, n=384):
    #r1 = TimeDistributed(Convolution2D(384, 3, 3, activation='relu', subsample=(2,2),border_mode='same'))(input)
    r1 = TimeDistributed(MaxPooling2D((3,3), strides=(2,2), border_mode='same'))(input)
    r2 = TimeDistributed(Convolution2D(n, 3, 3, activation='relu', subsample=(2,2),border_mode='same'))(input)

    r3 = TimeDistributed(Convolution2D(k, 1, 1, activation='relu', border_mode='same'))(input)
    r3 = TimeDistributed(Convolution2D(l, 3, 3, activation='relu', border_mode='same'))(r3)
    #m_pad =ZeroPadding2D((1,1))(r3)
    r3 = TimeDistributed(Convolution2D(m, 3, 3, activation='relu', subsample=(2,2),border_mode='same'))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis_5d)
    m = BatchNormalization(axis=2)(m)
    m = Activation('relu')(m)
    return m

def time_reduction_resnet_v2_B(input):
    r1 = TimeDistributed(MaxPooling2D((3,3), strides=(2,2), border_mode='same'))(input)
    #r1 = TimeDistributed(Convolution2D(1152, 3, 3, activation='relu', subsample=(2,2),border_mode='same'))(input)

    r2 = TimeDistributed(Convolution2D(256, 1, 1, activation='relu', border_mode='same'))(input)
    r2 = TimeDistributed(Convolution2D(384, 3, 3, activation='relu', subsample=(2,2),border_mode='same'))(r2)

    r3 = TimeDistributed(Convolution2D(256, 1, 1, activation='relu', border_mode='same'))(input)
    r3 = TimeDistributed(Convolution2D(288, 3, 3, activation='relu', subsample=(2, 2),border_mode='same'))(r3)

    r4 = TimeDistributed(Convolution2D(256, 1, 1, activation='relu', border_mode='same'))(input)
    r4 = TimeDistributed(Convolution2D(288, 3, 3, activation='relu', border_mode='same'))(r4)
    r4 = TimeDistributed(Convolution2D(320, 3, 3, activation='relu', subsample=(2, 2),border_mode='same'))(r4)

    m = merge([r1, r2, r3, r4], concat_axis=channel_axis_5d, mode='concat')
    m = BatchNormalization(axis=2)(m)
    m = Activation('relu')(m)
    return m


def model_test_deconv(input):
    x1 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(input)
    x2 = TimeDistributed(MaxPooling2D(pool_size=(8, 8)))(x1)
    #u1=  TimeDistributed(Deconvolution2D(2, 5, 5, output_shape=(None, 2, 256, 256), subsample=(2, 2), border_mode='same'))(x2)
    #u2 = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(u1)
   # u2 = ZeroPadding3D(padding=(0,1, 1))(u1)
    u1 = DeconvLSTM2D(nb_filter=2, nb_row=15, nb_col=15,go_backwards=False, subsample=(8,8), dim_ordering='th', border_mode='same',return_sequences=True)(x2)
    return u1



# This is DeepEM3D model, which used in the paper We have submitted To Bioinformatic
# Note that original dialted multi-scale decovlution has been replaced using sample UPscale operation for two reason:
# 1) the theano bacl currently do not support dialation conv, 2) dialted de-conv
# seems not very important for improve accuracy.
def timeDist_DeepEM3D_Net_simpleUP(input, scale=True):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x1,z1 = time_inception_resnet_stem_3Dconv(input)
    x1 = time_inception_resnet_v2_A(x1, scale_residual=scale)
    x2 = time_reduction_A(x1, k=256, l=256, m=384, n=384)
    x2 = time_inception_resnet_v2_B(x2, scale_residual=scale)
    x3 = time_reduction_resnet_v2_B(x2)
    x3 = time_inception_resnet_v2_C(x3, scale_residual=scale)
    u1 =TimeDistributed(UpSampling2D(size=(2, 2)))(z1)
    u1 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(u1)
    #u1= Bidirectional(ConvLSTM2D(nb_filter=2, nb_row=3, nb_col=3,
    #                       dim_ordering="th", border_mode="same", return_sequences=True), merge_mode='sum')(u1)
    u1 = BatchNormalization(axis=2)(u1)

    u2 =TimeDistributed(UpSampling2D(size=(4, 4)))(x1)
    #u2= Bidirectional(LSTMConv2D(nb_filter=2, nb_row=3, nb_col=3,
    #                       dim_ordering="th", border_mode="same", return_sequences=True), merge_mode='sum')(u2)
    u2 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(u2)
    u2 = BatchNormalization(axis=2)(u2)

    u3 =TimeDistributed(UpSampling2D(size=(8, 8)))(x2)
    #u3= Bidirectional(LSTMConv2D(nb_filter=2, nb_row=3, nb_col=3,
    #                       dim_ordering="th", border_mode="same", return_sequences=True), merge_mode='sum')(u3)
    u3 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(u3)
    u3 = BatchNormalization(axis=2)(u3)

    u4 =TimeDistributed(UpSampling2D(size=(16, 16)))(x3)
    #u4= Bidirectional(LSTMConv2D(nb_filter=2, nb_row=3, nb_col=3,
    #                       dim_ordering="th", border_mode="same", return_sequences=True), merge_mode='sum')(u4)

    u4 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(u4)
    u4 = BatchNormalization(axis=2)(u4)
    merged = merge([u1, u2,u3,u4], mode='sum')
    
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(merged)
    return out

# This is DeepEM3D model, which used in the paper We have submitted To Bioinformatic
# Note that original dialted multi-scale decovlution has been replaced by only one decovolution for each scale
def timeDist_DeepEM3D_Net(input, scale=True):
    # Input Shape is 256 x 256 x 3 (tf) or 3 x 256 x 256 (th)
    w =1024
    h=1024
    x1,z1 = time_inception_resnet_stem_3Dconv(input)
    x1 = time_inception_resnet_v2_A(x1, scale_residual=scale)
    x2 = time_reduction_A(x1, k=256, l=256, m=384, n=384)
    x2 = time_inception_resnet_v2_B(x2, scale_residual=scale)
    x3 = time_reduction_resnet_v2_B(x2)
    x3 = time_inception_resnet_v2_C(x3, scale_residual=scale)
    u1 =TimeDistributed(Deconvolution2D(2, 5, 5, output_shape=(None, 2, h, w), subsample=(2, 2), border_mode='same'))(z1)
    u1 = BatchNormalization(axis=2)(u1)

    u2 =TimeDistributed(Deconvolution2D(2, 9, 9, output_shape=(None, 2, h, w), subsample=(4, 4), border_mode='same'))(x1)
    u2 = BatchNormalization(axis=2)(u2)

    u3 =TimeDistributed(Deconvolution2D(2, 17, 17, output_shape=(None, 2, h, w), subsample=(8, 8), border_mode='same'))(x2)
    u3 = BatchNormalization(axis=2)(u3)

    u4 =TimeDistributed(Deconvolution2D(2, 33, 33, output_shape=(None, 2, h, w), subsample=(16, 16), border_mode='same'))(x3)
    u4 = BatchNormalization(axis=2)(u4)
    merged = merge([u1, u2,u3,u4], mode='sum')
    
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(merged)
    return out

def timeDist_DeepEM2D_Net(input, scale=True):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    w =256
    h=256
    x1,z1 = time_inception_resnet_stem_2Dconv(input)
    x1 = time_inception_resnet_v2_A(x1, scale_residual=scale)
    x2 = time_reduction_A(x1, k=256, l=256, m=384, n=384)
    x2 = time_inception_resnet_v2_B(x2, scale_residual=scale)
    x3 = time_reduction_resnet_v2_B(x2)
    x3 = time_inception_resnet_v2_C(x3, scale_residual=scale)
    u1 =TimeDistributed(Deconvolution2D(2, 5, 5, output_shape=(None, 2, h, w), subsample=(2, 2), border_mode='same'))(z1)
    u1 = BatchNormalization(axis=2)(u1)

    u2 =TimeDistributed(Deconvolution2D(2, 9, 9, output_shape=(None, 2, h, w), subsample=(4, 4), border_mode='same'))(x1)
    u2 = BatchNormalization(axis=2)(u2)

    u3 =TimeDistributed(Deconvolution2D(2, 17, 17, output_shape=(None, 2, h, w), subsample=(8, 8), border_mode='same'))(x2)
    u3 = BatchNormalization(axis=2)(u3)

    u4 =TimeDistributed(Deconvolution2D(2, 33, 33, output_shape=(None, 2, h, w), subsample=(16, 16), border_mode='same'))(x3)
    u4 = BatchNormalization(axis=2)(u4)
    merged = merge([u1, u2,u3,u4], mode='sum')
    
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(merged)
    return out

# This one remove the 3D conv and replace the Deconv layer with Deconv-GRU layer
def timeDist_DeepEM3D_Net_DeconvLSTM(input, scale=True):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x1,z1 = time_inception_resnet_stem_2Dconv(input)
    x1 = time_inception_resnet_v2_A(x1, scale_residual=scale)
    x2 = time_reduction_A(x1, k=256, l=256, m=384, n=384)
    x2 = time_inception_resnet_v2_B(x2, scale_residual=scale)
    x3 = time_reduction_resnet_v2_B(x2)
    x3 = time_inception_resnet_v2_C(x3, scale_residual=scale)
    

    u1=time_DeconvLSTM_bottleNeck_block(z1,2,5,5,subsample=(2,2))

    #u1 =TimeDistributed(UpSampling2D(size=(2, 2)))(z1)
    #u1 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(u1)
    #u1= Bidirectional(ConvLSTM2D(nb_filter=2, nb_row=3, nb_col=3,
    #                       dim_ordering="th", border_mode="same", return_sequences=True), merge_mode='sum')(u1)
    u1 = BatchNormalization(axis=2)(u1)


    u2=time_DeconvLSTM_bottleNeck_block(x1,2,9,9,subsample=(4,4))

    #u2 =TimeDistributed(UpSampling2D(size=(4, 4)))(x1)
    #u2= Bidirectional(LSTMConv2D(nb_filter=2, nb_row=3, nb_col=3,
    #                       dim_ordering="th", border_mode="same", return_sequences=True), merge_mode='sum')(u2)
    #u2 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(u2)
    u2 = BatchNormalization(axis=2)(u2)


    u3=time_DeconvLSTM_bottleNeck_block(x2,2,17,17,subsample=(8,8))
    #u3 =TimeDistributed(UpSampling2D(size=(8, 8)))(x2)
    #u3= Bidirectional(LSTMConv2D(nb_filter=2, nb_row=3, nb_col=3,
    #                       dim_ordering="th", border_mode="same", return_sequences=True), merge_mode='sum')(u3)
    #u3 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(u3)
    u3 = BatchNormalization(axis=2)(u3)

    u4=time_DeconvLSTM_bottleNeck_block(x3,2,33,33,subsample=(16,16))
    #4 =TimeDistributed(UpSampling2D(size=(16, 16)))(x3)
    #u4= Bidirectional(LSTMConv2D(nb_filter=2, nb_row=3, nb_col=3,
    #                       dim_ordering="th", border_mode="same", return_sequences=True), merge_mode='sum')(u4)

    #u4 = TimeDistributed(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))(u4)
    u4 = BatchNormalization(axis=2)(u4)
    merged = merge([u1, u2,u3,u4], mode='sum')
    #out=Activation('sigmoid')(merged)
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(merged)
    return out


def timeDist_DeepEM3D_Net_DeconvGRU(input, scale=True):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x1,z1 = time_inception_resnet_stem_2Dconv(input)
    x1 = time_inception_resnet_v2_A(x1, scale_residual=scale)
    x2 = time_reduction_A(x1, k=256, l=256, m=384, n=384)
    x2 = time_inception_resnet_v2_B(x2, scale_residual=scale)
    x3 = time_reduction_resnet_v2_B(x2)
    x3 = time_inception_resnet_v2_C(x3, scale_residual=scale)
    

    u1=time_DeconvGRU_bottleNeck_block(z1,2,5,5,subsample=(2,2))
    u1 = BatchNormalization(axis=2)(u1)


    u2=time_DeconvGRU_bottleNeck_block(x1,2,9,9,subsample=(4,4))
    u2 = BatchNormalization(axis=2)(u2)


    u3=time_DeconvGRU_bottleNeck_block(x2,2,17,17,subsample=(8,8))
    u3 = BatchNormalization(axis=2)(u3)

    u4=time_DeconvGRU_bottleNeck_block(x3,2,33,33,subsample=(16,16))
    u4 = BatchNormalization(axis=2)(u4)
    merged = merge([u1, u2,u3,u4], mode='sum')
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(merged)
    return out




    ## ----------------======================= UNet Varaints =================== ------------------#
def time_unet2D(inputs):
    #inputs = Input((1, img_rows, img_cols))
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)

    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=2)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)

    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=2)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=2)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=2)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv10 = TimeDistributed(Convolution2D(2, 1, 1, activation='relu'))(conv9)
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(conv10)
    return out


def time_GRU_unet_1_level(inputs):
    #inputs = Input((1, img_rows, img_cols))
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = time_ConvGRU_bottleNeck_block(conv5,512,3,3)
    #conv5_1 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=2)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)


    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=2)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=2)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=2)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv10 = TimeDistributed(Convolution2D(2, 1, 1, activation='relu'))(conv9)
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(conv10)
    return out


def time_GRU_unet_3_level(inputs):
    #inputs = Input((1, img_rows, img_cols))
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    #conv3 = time_ConvGRU_bottleNeck_block(conv3,128,3,3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    #conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    conv4 = time_ConvGRU_bottleNeck_block(conv4,256,3,3)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = time_ConvGRU_bottleNeck_block(conv5,512,3,3)
    #conv5_1 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=2)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = time_ConvGRU_bottleNeck_block(conv6,256,3,3)
    #conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)


    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=2)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    #conv7 = time_ConvGRU_bottleNeck_block(conv7,128,3,3)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=2)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=2)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv10 = TimeDistributed(Convolution2D(2, 1, 1, activation='relu'))(conv9)
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(conv10)
    return out


def time_GRU_unet_5_level(inputs):
    #inputs = Input((1, img_rows, img_cols))
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    #conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    conv3 = time_ConvGRU_bottleNeck_block(conv3,128,3,3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    #conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    conv4 = time_ConvGRU_bottleNeck_block(conv4,256,3,3)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = time_ConvGRU_bottleNeck_block(conv5,512,3,3)
    #conv5_1 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=2)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = time_ConvGRU_bottleNeck_block(conv6,256,3,3)
    #conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)


    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=2)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    conv7 = time_ConvGRU_bottleNeck_block(conv7,128,3,3)
    #conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=2)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=2)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv10 = TimeDistributed(Convolution2D(2, 1, 1, activation='relu'))(conv9)
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(conv10)
    return out



def time_LSTM_unet_1_level(inputs):
    #inputs = Input((1, img_rows, img_cols))
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = time_ConvLSTM_bottleNeck_block(conv5,512,3,3)
    #conv5_1 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=2)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)


    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=2)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=2)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=2)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv10 = TimeDistributed(Convolution2D(2, 1, 1, activation='relu'))(conv9)
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(conv10)
    return out

def time_LSTM_unet_5_level(inputs):
    #inputs = Input((1, img_rows, img_cols))
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    #conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    conv3 = time_ConvLSTM_bottleNeck_block(conv3,128,3,3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    #conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    conv4 = time_ConvLSTM_bottleNeck_block(conv4,256,3,3)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = time_ConvLSTM_bottleNeck_block(conv5,512,3,3)
    #conv5_1 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=2)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = time_ConvLSTM_bottleNeck_block(conv6,256,3,3)
    #conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)


    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=2)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    conv7 = time_ConvLSTM_bottleNeck_block(conv7,128,3,3)
    #conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=2)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=2)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv10 = TimeDistributed(Convolution2D(2, 1, 1, activation='relu'))(conv9)
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(conv10)
    return out

def time_LSTM_unet_3_level(inputs):
    #inputs = Input((1, img_rows, img_cols))
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    #conv3 = time_ConvLSTM_bottleNeck_block(conv3,128,3,3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    #conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    conv4 = time_ConvLSTM_bottleNeck_block(conv4,256,3,3)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = time_ConvLSTM_bottleNeck_block(conv5,512,3,3)
    #conv5_1 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=2)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = time_ConvLSTM_bottleNeck_block(conv6,256,3,3)
    #conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)


    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=2)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    #conv7 = time_ConvLSTM_bottleNeck_block(conv7,128,3,3)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=2)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=2)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv10 = TimeDistributed(Convolution2D(2, 1, 1, activation='relu'))(conv9)
    out=Lambda(time_dist_softmax,output_shape=time_dist_softmax_out_shape)(conv10)
    return out

