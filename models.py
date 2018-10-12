from keras.models import Model
import keras.backend as K
from keras import regularizers
from keras.engine.topology import Layer
from keras.layers.core import Reshape, Permute
from keras.layers import Dropout, merge, Input, Dense, Flatten, Conv2D,GlobalAveragePooling2D, Concatenate,MaxPooling2D, GlobalMaxPooling2D,ZeroPadding2D,ZeroPadding1D
K.set_image_dim_ordering('tf')

shared = 1
depth = 5
attention = 1


'''
For 2 Features call ensemble1 function
for 3 features call ensemble function
 
'''
def ensemble1(dimx0,dimx1,dimy0,dimy1,num_classes,**kwargs):

  input_neurons  = kwargs['kwargs'].get('input_neurons',100)
  act1           = kwargs['kwargs'].get('act1','relu')
  act2           = kwargs['kwargs'].get('act2','sigmoid')
  act3           = kwargs['kwargs'].get('act3','softmax')
  act4           = kwargs['kwargs'].get('act4','tanh')
  nb_filter      = kwargs['kwargs'].get('nb_filter',200)
  frame_size     = kwargs['kwargs'].get('frame_size',10)
  filter_length  = kwargs['kwargs'].get('filter_length',3)
  pool_size      = kwargs['kwargs'].get('pool_size',(1,1))
  dropout        = kwargs['kwargs'].get('dropout',0.3)
  reg            = kwargs['kwargs'].get('reg',0.005)
  loss           = kwargs['kwargs'].get('loss','binary_crossentropy')
  optimizer      = kwargs['kwargs'].get('optimizer','adam')
  metrics        = kwargs['kwargs'].get('metrics','mse')

  if type(filter_length) is int:
    filter_length = [filter_length] * 2
  
  if act1==None or act2==None:
    print "2 or 3 Activations Required"
    return
  print "Activation 1 {} 2 {} 3 {} ".format(act1,act2,act3)
  print "Model CNN1"
  if attention:
    print 'with attention'
  else:
    print 'without attention'
  inpx0 = Input(shape=(dimx0,dimy0),name='inpx0')
  inpx1 = Input(shape=(dimx1,dimy1),name='inpx1')
  mul = MatchScore(inpx0,inpx1)
  mulT = Permute((2,1))(mul)

  d1_1 = Dense(units = frame_size)(mul)
  d2_1 = Dense(units = frame_size)(mulT)

  x0 = Permute((2,1))(inpx0)
  x1 = Permute((2,1))(inpx1)

  x0 = Reshape(( x0._keras_shape[1], x0._keras_shape[2], 1))(x0)
  x1 = Reshape(( x1._keras_shape[1], x1._keras_shape[2], 1))(x1)
 
  d1_1 = Reshape(( d1_1._keras_shape[2], d1_1._keras_shape[1], 1))(d1_1)
  d2_1 = Reshape(( d2_1._keras_shape[2], d2_1._keras_shape[1], 1))(d2_1)

  

  if attention in [1,3]:    
    conv1 = merge([x0,d1_1],mode='concat',concat_axis=3)
    conv2 = merge([x1,d2_1],mode='concat',concat_axis=3)
  else:
    conv1,conv2 = x0, x1

  channel_1, channel_2 = [], []


  
  filter_len = [60,3]
  input_neurons = 600
  filter_len_x = (filter_len[0],filter_len[1])
  filter_len_y = (filter_len[0],filter_len[1])
  depth = 6

  for dep in range(depth):

    if shared:
        conv = Conv2D(nb_filter=256, kernel_size = filter_len_x, activation=act1,
                    data_format = 'channels_last',border_mode="valid")
        ques = conv(conv1)
        ans = conv(conv2)
        
        print ques._keras_shape
        print ans._keras_shape
        if depth<2:
            
            conv_1 = Conv2D(nb_filter=128, kernel_size = filter_len_x, activation=act4,
                        data_format = 'channels_last',border_mode="valid")
            ques_1 = conv_1(ques)
            ans_1 = conv_1(ans)
            
            conv_2 = Conv2D(nb_filter=128, kernel_size = filter_len_x, activation=act4,
                        data_format = 'channels_last',border_mode="valid")
            ques= conv_2(ques_1)
            ans= conv_2(ans_1)
        

    else:
        ques = Conv2D(nb_filter=nb_filter, kernel_size = filter_len_x, activation=act4,
                data_format = 'channels_last',border_mode="valid")(conv1)
        ans = Conv2D(nb_filter, kernel_size = filter_len_y, activation=act4,
                data_format="channels_last",border_mode="valid")(conv2)


    ques = Dropout(dropout)(ques)
    ans = Dropout(dropout)(ans)
    

    conv1 = MaxPooling2D(pool_size)(ques)
    conv2 = MaxPooling2D(pool_size)(ans)

    channel_1.append(GlobalAveragePooling2D()(ques))
    channel_2.append(GlobalAveragePooling2D()(ans))

    filter_len_x = (conv1._keras_shape[-3],filter_len[1])
    filter_len_y = (conv2._keras_shape[-3],filter_len[1])



  h1 = channel_1.pop(-1)
  if channel_1:
    h1 = merge([h1] + channel_1, mode="concat")

  h2 = channel_2.pop(-1)
  if channel_2:
    h2 = merge([h2] + channel_2, mode="concat")

  h =  Concatenate(name='h')([h1, h2])

  h = Dense(input_neurons,activation='relu',kernel_regularizer=regularizers.l2(reg))(h)
  score = Dense(num_classes,activation='sigmoid',name='score')(h)
  model = Model(input=([inpx0, inpx1]),output= score)
  model.summary()
  model.compile( loss= loss,optimizer= optimizer ,metrics=[metrics])

  return model

def ensemble(dimx0,dimx1,dimy0,dimy1,dimx2,dimy2,num_classes,**kwargs):
    
    input_neurons  = kwargs['kwargs'].get('input_neurons',100)
    act1           = kwargs['kwargs'].get('act1','relu')
    act2           = kwargs['kwargs'].get('act2','sigmoid')
    act3           = kwargs['kwargs'].get('act3','softmax')
    act4           = kwargs['kwargs'].get('act4','tanh')
    nb_filter      = kwargs['kwargs'].get('nb_filter',100)
    frame_size     = kwargs['kwargs'].get('frame_size',80)
    filter_length  = kwargs['kwargs'].get('filter_length',3)
    pool_size      = kwargs['kwargs'].get('pool_size',(1,1))
    dropout        = kwargs['kwargs'].get('dropout',0.3)
    reg            = kwargs['kwargs'].get('reg',0.005)
    loss           = kwargs['kwargs'].get('loss','binary_crossentropy')
    optimizer      = kwargs['kwargs'].get('optimizer','adam')
    metrics        = kwargs['kwargs'].get('metrics','mse')
    
    if type(filter_length) is int:
        filter_length = [filter_length] * 2

    if act1==None or act2==None:
        print "2 or 3 Activations Required"
        return
    print "Activation 1 {} 2 {} 3 {} ".format(act1,act2,act3)
    print "Model CNN1"
    if attention:
        print 'with attention'
    else:
        print 'without attention'

    inpx0 = Input(shape=(dimx0,dimy0),name='inpx0')
    inpx1 = Input(shape=(dimx1,dimy1),name='inpx1')
    inpx2 = Input(shape=(dimx2,dimy2),name='inpx2')
    
    x0,x1,x2 = inpx0,inpx1,inpx2
    mul = MatchScore(x0,x1)
    mulT = Permute((2,1))(mul)
    
    all_dense1 = Dense(units = frame_size)
    all_dense2 = Dense(units = frame_size)
    
    all_dense2 = all_dense1

    d1_1 = all_dense1(mul)
    d2_1 = all_dense1(mulT)
       
    mul = MatchScore(x1,x2)
    mulT = Permute((2,1))(mul)

    d1_2 = all_dense2(mul)
    d2_2 = all_dense2(mulT)
    
    
    x0 = Permute((2,1))(inpx0)
    x1 = Permute((2,1))(inpx1)
    x2 = Permute((2,1))(inpx2)

    
    x0 = Reshape(( x0._keras_shape[1], x0._keras_shape[2], 1))(x0)
    x1 = Reshape(( x1._keras_shape[1], x1._keras_shape[2], 1))(x1)
    x2 = Reshape(( x2._keras_shape[1], x2._keras_shape[2], 1))(x2)

    d1_1 = Reshape(( d1_1._keras_shape[2], d1_1._keras_shape[1], 1))(d1_1)
    d2_1 = Reshape(( d2_1._keras_shape[2], d2_1._keras_shape[1], 1))(d2_1)
    
    d1_2 = Reshape(( d1_2._keras_shape[2], d1_2._keras_shape[1], 1))(d1_2)
    d2_2 = Reshape(( d2_2._keras_shape[2], d2_2._keras_shape[1], 1))(d2_2)
    
    conv1 = merge([x0,d1_1],mode='concat',concat_axis=3)
    conv2 = merge([x1,d2_1],mode='concat',concat_axis=3)
    conv3 = merge([x2,d2_2],mode='concat',concat_axis=3)
    
    channel_1, channel_2 ,channel_3 = [], [], []
    
    filter_len = [60,3]
    input_neurons = 600
    filter_len_x = (filter_len[0],filter_len[1])
    filter_len_y = (filter_len[0],filter_len[1])
    filter_len_z = (filter_len[0],filter_len[1])
    depth = 6
    
    for dep in range(depth):
        
        if shared:
            conv = Conv2D(nb_filter=256, kernel_size = filter_len_x, activation=act4,
                        data_format = 'channels_last',border_mode="valid")           
            aud_f1 = conv(conv1)
            aud_f2 = conv(conv2)
            aud_f3 = conv(conv3)
            
            if depth<2:
                print('Hello')
                conv_1 = Conv2D(nb_filter=256, kernel_size = filter_len_x, activation=act4,
                        data_format = 'channels_last',border_mode="valid")           
                aud_f1 = conv_1(aud_f1)
                aud_f2 = conv_1(aud_f2)
                aud_f3 = conv_1(aud_f3)
                
                print(aud_f1._keras_shape)
                print(aud_f2._keras_shape)
                print(aud_f3._keras_shape)
                
                conv_2 = Conv2D(nb_filter=256, kernel_size = filter_len_x, activation=act4,
                        data_format = 'channels_last',border_mode="valid")           
                aud_f1 = conv_2(aud_f1)
                aud_f2 = conv_2(aud_f2)
                aud_f3 = conv_2(aud_f3)
                
                
        else:
            aud_f1 = Conv2D(nb_filter=nb_filter, kernel_size = filter_len_x, activation=act4,
                    data_format = 'channels_last',padding='same')(conv1)
            aud_f2 = Conv2D(nb_filter=nb_filter, kernel_size = filter_len_y, activation=act4,
                    data_format = 'channels_last',padding='same')(conv2)
            aud_f3 = Conv2D(nb_filter=nb_filter, kernel_size = filter_len_z, activation=act4,
                    data_format = 'channels_last',padding='same')(conv3)
            
        aud_f1 = Dropout(dropout)(aud_f1)
        aud_f2 = Dropout(dropout)(aud_f2)
        aud_f3 = Dropout(dropout)(aud_f3)
     
        conv1 = MaxPooling2D(pool_size)(aud_f1)
        conv2 = MaxPooling2D(pool_size)(aud_f2)
        conv3 = MaxPooling2D(pool_size)(aud_f3)
        
        
        channel_1.append(GlobalAveragePooling2D()(aud_f1))
        channel_2.append(GlobalAveragePooling2D()(aud_f2))
        channel_3.append(GlobalAveragePooling2D()(aud_f3))
        
        filter_len_x = (conv1._keras_shape[-3],filter_len[1])
        filter_len_y = (conv2._keras_shape[-3],filter_len[1])
        filter_len_z = (conv3._keras_shape[-3],filter_len[1])

    h1 = channel_1.pop(-1)
    if channel_1:
        h1 = merge([h1] + channel_1, mode="concat")

    h2 = channel_2.pop(-1)
    if channel_2:
        h2 = merge([h2] + channel_2, mode="concat")
        
    h3 = channel_3.pop(-1)
    if channel_3:
        h3 = merge([h3] + channel_3, mode="concat")
        
    print (h1._keras_shape)
    print (h2._keras_shape)
    print (h3._keras_shape)
        
    sim = Similarity(h1._keras_shape[1])
    
    sim1 = sim([h1,h2])
    sim2 = sim([h2,h3])
    
    h =  Concatenate(name='h')([h1, h2, h3, sim1, sim2])
    
    h = Dense(input_neurons,activation='relu',kernel_regularizer=regularizers.l2(reg))(h)
    score = Dense(num_classes,activation='sigmoid',name='score')(h)
    
    model = Model([inpx0, inpx1, inpx2],[score])
    model.summary()
    model.compile( loss=loss,optimizer=optimizer,metrics=[metrics])

    return model

def compute_euclidean_match_score(l_r):
    l, r = l_r
    denominator = 1. + K.sqrt(
        -2 * K.batch_dot(l, r, axes=[2, 2]) +
        K.expand_dims(K.sum(K.square(l), axis=2), 2) +
        K.expand_dims(K.sum(K.square(r), axis=2), 1)
    )
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator

def MatchScore(l, r, mode="euclidean"):
    if mode == "euclidean":
        return merge(
            [l, r],
            mode=compute_euclidean_match_score,
            output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
        )
        
        
class Similarity(Layer):
    
    def __init__(self, v_dim, kernel_regularizer=None, **kwargs):
        self.v_dim = v_dim
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(Similarity, self).__init__(**kwargs)

    def build(self,input_shape):
        self.W = self.add_weight(name='w',shape=(self.v_dim, self.v_dim),
                                      initializer='glorot_uniform',
                                      regularizer=self.kernel_regularizer,
                                  trainable=True)     
                                
        super(Similarity, self).build(input_shape)

    def call(self, data, mask=None):
        v1 = data[0]
        v2 = data[1]
        sim = K.dot(v1,self.W)
        sim = K.batch_dot(sim,v2,axes=1)
        return sim

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)