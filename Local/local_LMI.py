import sys
import ROOT
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_addons.activations import tanhshrink
sys.path.append('../../')
from Formulation.BHDVCS_tf_modified import TotalFLayer
import matplotlib.pyplot as plt
import time

tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

def np_data(set):
    set_cut = "set == " + str(set)
    df = ROOT.RDF.FromCSV('/media/lily/Data/GPDs/DVCS/Pseudodata/JLabKinematics/withRealErrors/pseudo_'+GPD_MODEL+'_BKM10_Jlab_all_t2.csv')
    npset = df.Filter(set_cut).AsNumpy() 
    return npset

def gen_replica(pseudo):
    F_rep = np.random.normal(loc=pseudo['F'], scale=abs(pseudo['varF']*pseudo['F'])) # added abs for runtime error: 'ValueError: scale < 0'
    errF_rep = pseudo['varF'] * F_rep
    
    replica_dict = {'set': pseudo['set'],
                    'k': pseudo['k'],
                    'QQ': pseudo['QQ'],
                    'xB': pseudo['xB'],
                    't': pseudo['t'], 
                    'phi': pseudo['phi'], 
                    'F': F_rep,
                    'errF': errF_rep}   
     
    return replica_dict

def build_model():
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=123)  
    inputs = tf.keras.Input(shape=(5)) # k, QQ, x_b, t, phi
    k, QQ, xB, t, phi = tf.split(inputs, num_or_size_splits=5, axis=1)
    # Normalization
    QQ_norm = tf.keras.layers.Lambda(lambda x: -1 + 2 * (x / 10))(QQ) 
    xB_norm = tf.keras.layers.Lambda(lambda x: -1 + 2 * (x / 0.8))(xB) 
    t_norm = tf.keras.layers.Lambda(lambda x:  -1 + 2 * ((x + 2) / 2 ))(t) 
    # Concatenate
    kinematics = tf.keras.layers.concatenate([QQ_norm, xB_norm, t_norm], axis=1)
    x1 = tf.keras.layers.Dense(100, activation="linear", kernel_initializer=initializer)(kinematics)
    x2 = tf.keras.layers.Dense(100, activation="tanhshrink", kernel_initializer=initializer)(x1)
    x3 = tf.keras.layers.Dense(100, activation="tanhshrink", kernel_initializer=initializer)(x2)
    x4 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(x3)
    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x4)
    #### k, QQ, xB, t, phi, ReH, ReE, ReHt, dvcs ####
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2
    #tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name="tfmodel")
    tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF)
    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(0.00025),
        loss = tf.keras.losses.MeanSquaredError()
    )
    return tfModel

def fit_replica(i, pseudo):
    # ---- generate replica -----
    replica = gen_replica(pseudo)
    kin = np.dstack((replica['k'], replica['QQ'],replica['xB'], replica['t'], replica['phi']))
    kin = kin.reshape(kin.shape[1:])
    # ---- model fit ---- 
    kin_train, kin_test, F_train, F_test = train_test_split(kin, replica['F'], test_size=0.1, random_state=42)
    # Create a callback to stop training early after reaching a certain value for the validation loss.
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, verbose=1, mode='auto') # used it for models 6 with patience = 10
    models = build_model()
    models.summary()
    history = models.fit(kin_train, F_train, epochs=150, batch_size=1, validation_data=(kin_test, F_test))
    tf.keras.models.save_model(models, 'models/'+GPD_MODEL+'/batch_300/fit_replica_'+str(i)+'.keras') # need "tf.keras.models.save_model" to save custom layer
    np.save('models/'+GPD_MODEL+'/batch_300/history_fit_replica_'+str(i)+'.npy',history.history) 


NUM_OF_REPLICAS = 100
GPD_MODEL = 'basic'
SET_NUM = 1

pseudo = np_data(SET_NUM)   

for i in range(0, NUM_OF_REPLICAS):
    start = time.time()
    fit_replica(i, pseudo)
    print("Run Time: ", time.time() - start)
