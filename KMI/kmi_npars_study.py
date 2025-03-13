import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import RepeatedKFold, train_test_split
# from tensorflow_addons.optimizers import AdamW
sys.path.append('../')
from Formulation.BKM_DVCS import BKM_DVCS
import matplotlib.pyplot as plt
import time

@tf.function
def tanhshrink(x):
    return x - tf.math.tanh(x)

# tf.keras.utils.get_custom_objects().update({'tanhshrink': tf.keras.layers.Activation(tanhshrink)})
# tf.keras.utils.get_custom_objects().clear()
# tf.keras.utils.get_custom_objects()['tanhshrink'] = tf.keras.layers.Activation(tanhshrink)

bkm10 = BKM_DVCS()

GPD_MODEL = 'KM15'
NUM_OF_REPLICAS = 1
early_stop = False
replica = False
cross_validation = True

# datafile = '/Users/liliet/Library/CloudStorage/OneDrive-LosAlamosNationalLaboratory/R-KMI/Pseudodata/pseudo_'+GPD_MODEL+'_BKM10_Jlab_all_t2.csv'
datafile = '/Users/liliet/Library/CloudStorage/OneDrive-LosAlamosNationalLaboratory/R-KMI/Liliet/Pseudodata/pseudo_'+GPD_MODEL+'_BKM10_Jlab_all_t2_8pars.csv'

epochs = 1

# get (pseudo)data file
def get_data():
    df = pd.read_csv(datafile, dtype=np.float32)
    return df

# Normalize QQ, xB, t
def normalize(QQ, xB, t):
    QQ_norm = -1 + 2 * (QQ / 10) 
    xB_norm = -1 + 2 * (xB / 0.8)
    t_norm = -1 + 2 * ((t + 2) / 2 )
    return QQ_norm, xB_norm, t_norm

def gen_replica(pseudo):
    F_rep = np.random.normal(loc=pseudo['F'], scale=abs(pseudo['varF']*pseudo['F'])) # added abs for runtime error: 'ValueError: scale < 0'
    errF_rep = pseudo['varF'] * F_rep
    
    replica_dict = {'set': pseudo['set'], 
                    'k': pseudo['k'], 'QQ': pseudo['QQ'], 'xB': pseudo['xB'], 't': pseudo['t'],     
                    'phi': pseudo['phi'], 'F': F_rep,'errF': errF_rep}       
    return replica_dict

def build_model(npars):
    # model 75, info path: /media/lily/Data/GPDs/ANN/KMI/tunning/kt_RS
    model = tf.keras.Sequential([
        # model input train size all:  10458
        # model each input train size:  3486
        tf.keras.layers.Dense(3486,activation='relu6', input_shape=(3,)), 
        tf.keras.layers.Dense(2000, activation='relu6'),
        tf.keras.layers.Dense(1000, activation='relu6'),
        tf.keras.layers.Dense(500, activation='relu6'),
        # tf.keras.layers.Dense(100, activation='elu'),
        # tf.keras.layers.Dense(100, activation="tanh")),
        # tf.keras.layers.Dense(50, tf.keras.layers.Activation(tanhshrink)),
        tf.keras.layers.Dense(npars, activation="linear") # ReH, ReE, ReHt, dvcs
    ])   
    return model

# Reduced chi2 custom Loss function (model predicted inside loss)
def rchi2_Loss(kin, pars, F_data, F_err, option):
    kin = tf.cast(kin, pars.dtype)
    F_dnn = tf.reshape(bkm10.total_xs(kin, pars, "t2", option), [-1])
    F_data = tf.cast(F_data, pars.dtype)
    F_err = tf.cast(F_err, pars.dtype)
    loss = tf.reduce_mean(tf.square( (F_dnn - F_data) / (F_err) ) )
    return loss

def fit_replica(i, pseudo):
    # ----- prepare input data -----------  
    if replica:        
        data = gen_replica(pseudo) # generate replica
    else:
        data = pseudo  

    kin = np.dstack((data['k'], data['QQ'] , data['xB'], data['t'], data['phi']))
    kin = kin.reshape(kin.shape[1:]) # loss inputs
    QQ_norm, xB_norm, t_norm = normalize(data['QQ'] , data['xB'], data['t']) 
    kin3_norm = np.array([QQ_norm, xB_norm, t_norm]).transpose() # model inputs
    # pars_true = np.array([pseudo['ReH'], pseudo['ReE'], pseudo['ReHtilde'], pseudo['dvcs']]).transpose() # true parameters
    pars_true = np.array([pseudo['ReH'], pseudo['ReE'], pseudo['ReHt'], pseudo['dvcs']]).transpose() # true parameters
    # pars_true = np.array([pseudo['ReH'], pseudo['ReE'], pseudo['ReHt'], pseudo['ReEt'], pseudo['ImH'], pseudo['ImE'], pseudo['ImHt'], pseudo['ImEt']]).transpose() # true parameters
    # ---- split train and testing replica data samples ---- 
    if cross_validation:
        rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        for train_index, test_index in rkf.split(kin):
            kin_train, kin_test, kin3_norm_train, kin3_norm_test = kin[train_index], kin[test_index], kin3_norm[train_index], kin3_norm[test_index]
            F_train, F_test = data['F'][train_index], data['F'][test_index]
            Ferr_train, Ferr_test = data['errF'][train_index], data['errF'][test_index]
    else:
        kin_train, kin_test, kin3_norm_train, kin3_norm_test, F_train, F_test, Ferr_train, Ferr_test = train_test_split(kin, kin3_norm, data['F'], data['errF'], test_size=0.10, random_state=42)

    print('model input train size norm: ',kin_train.size)
    print('model each input train size norm: ',kin_train.shape[0])
    print('model input train size all: ',kin3_norm_train.size)
    print('model each input train size: ',kin3_norm_train.shape[0])


    
    model_1 = build_model(4)
    model_1.summary()
    tf.keras.utils.plot_model(model_1, to_file='./model_test/model_plot.pdf', show_shapes=True, show_layer_names=True, show_layer_activations=True)

    # Instantiate an optimizer to train the model.
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.00008)
    rmse_1 = tf.keras.metrics.RootMeanSquaredError()
    mape_1 = tf.keras.metrics.MeanAbsolutePercentageError()

    @tf.function
    def train_step_1(loss_inputs, inputs, targets, weights, option):
        with tf.GradientTape() as tape:
            pars = model_1(inputs)
            loss_value = rchi2_Loss(loss_inputs, pars, targets, weights, option)
        grads = tape.gradient(loss_value, model_1.trainable_weights)
        optimizer_1.apply_gradients(zip(grads, model_1.trainable_weights))
        return loss_value
        
    @tf.function
    def test_step_1(loss_inputs, inputs, targets, weights, option):
        pars = model_1(inputs)
        val_loss_value = rchi2_Loss(loss_inputs, pars, targets, weights, option)
        return val_loss_value
    
    # Functions to update the metrics
    # MAPE for a given parameter: accuracy = (100 - MAPE)
    @tf.function
    def metricWrapper_1(m, kin3_norm, pars_true):
        mape_1.reset_state()
        def mapeMetric_1():
            pars = model_1(kin3_norm)       
            mape_1.update_state(pars_true[:, m], pars[:, m])
            return tf.convert_to_tensor(mape_1.result(), np.float32)
        return mapeMetric_1()
    
    @tf.function
    def dvcs_metricWrapper_1(kin3_norm):
        mape_1.reset_state()
        # get true dvcs
        dvcs_true = np.array([pseudo['dvcs']]).transpose() 
        # get DNN dvcs
        pars = model_1(kin3_norm)  
        dvcs_dnn = pars[:, 3]
        def dvcs_mapeMetric_1():
            mape_1.update_state(dvcs_true, dvcs_dnn)
            return tf.convert_to_tensor(mape_1.result(), np.float32)
        return dvcs_mapeMetric_1()
    
    # F RMSE weighted over F_errors
    @tf.function
    def rmseMetric_1(kin, kin3_norm, pars_true, F_errors, option):    
        pars = model_1(kin3_norm)
        kin = tf.cast(kin, pars.dtype)        
        pars_true = tf.cast(pars_true, pars.dtype)
        F_dnn = tf.reshape(bkm10.total_xs(kin, pars, "t2", option), [-1])
        F_true = np.array([pseudo['F_true']]).transpose() 
        weights = 1. / F_errors
        rmse_1.update_state(F_true, F_dnn, sample_weight = weights)
        return tf.convert_to_tensor(rmse_1.result(), np.float32)
    
    # -----------------------------------------------
    # Training with the 4th paramater being the dvcs 
    # -----------------------------------------------
 
    # Keep results for plotting
    train_loss_results = []
    val_loss_results = []
    F_rmse_results = []
    total_mape_results = []
    ReH_mape_results = []
    ReE_mape_results = []
    ReHt_mape_results = []
    dvcs_mape_results = []
    predictions_results = []

    patience = 50
    wait = 0
    best = float("inf")

    option = "constant"
    
    for epoch in range(epochs):
       
        loss_value = train_step_1(kin_train, kin3_norm_train, F_train, Ferr_train, option)
        val_loss_value = test_step_1(kin_test, kin3_norm_test, F_test, Ferr_test, option)

        # Update metrics    
        F_rmse = rmseMetric_1(kin, kin3_norm, pars_true, pseudo['errF'], option)
        pars_mape = [metricWrapper_1(m, kin3_norm, pars_true).numpy()  for m in range(3)]
        pars_mape = np.append(pars_mape, dvcs_metricWrapper_1(kin3_norm))
        total_mape = np.mean(pars_mape) 
         
        # End epoch
        train_loss_results.append(loss_value)
        val_loss_results.append(val_loss_value)
        F_rmse_results.append(F_rmse)
        total_mape_results.append(total_mape)
        ReH_mape_results.append(pars_mape[0])
        ReE_mape_results.append(pars_mape[1])
        ReHt_mape_results.append(pars_mape[2])
        dvcs_mape_results.append(pars_mape[3])
        print("Epoch {:03d}: Loss: {:.3f} val_Loss: {:.3f} F_rmse: {:.5f} ReH_mape: {:.5f} ReE_mape: {:.5f} ReHt_mape: {:.5f} dvcs_mape: {:.5f} total_mape: {:.5f}".format(epoch, loss_value, val_loss_value, F_rmse, pars_mape[0], pars_mape[1], pars_mape[2], pars_mape[3], total_mape))
        # print("Epoch {:03d}: Loss: {:.3f} val_Loss: {:.3f} F_rmse: {:.5f} ReH_mape: {:.5f} ReE_mape: {:.5f} ReHt_mape: {:.5f} total_mape: {:.5f}".format(epoch, loss_value, val_loss_value, F_rmse, pars_mape[0], pars_mape[1], pars_mape[2], total_mape))

        # Reset training metrics at the end of each epoch
        rmse_1.reset_state()
        mape_1.reset_state()

        # Get prediction for one set (set 1) for visualization
        if epoch % 10 == 0:
            # predictions = model(kin3_norm[:1])
            predictions = model_1.predict(kin3_norm[:1])
            predictions_results.append(predictions[:1])

        # Apply the early stopping strategy after 1000 epochs: stop the training if `total_mape` does not
        # decrease over a certain number of epochs.
        if early_stop:
            if epoch > 1000:
                wait += 1
                if total_mape < best:
                    best = total_mape
                    wait = 0
                if wait >= patience:
                    print(f'\nEarly stopping. No improvement in average MAPE in the last {patience} epochs.')
                    break
    # history = {'loss': train_loss_results, 'val_loss': val_loss_results, 'ReH_mape': ReH_mape_results, 'ReE_mape': ReE_mape_results, 'ReHt_mape': ReHt_mape_results, 'dvcs_mape': dvcs_mape_results, 'total_mape': total_mape_results}
    # tf.keras.models.save_model(model, 'models/'+GPD_MODEL+'/test/fit_replica_'+str(i)+'.keras') # need "tf.keras.models.save_model" to save custom layer
    # np.save('models/'+GPD_MODEL+'/test/history_fit_replica_'+str(i)+'.npy',history) 

    predictions_results = np.array(predictions_results)

    # -------------------------------------------------------------------------------------------------
    # Training with the 4th parameter being the dvcs coefficient term not depending on ReH, ReE and ReHt
    # ------------------------------------------------------------------------------------------------

    model_2 = build_model(4)
    model_2.summary()

    # Instantiate an optimizer to train the model.
    optimizer_2 = tf.keras.optimizers.Adam(learning_rate=0.0001)
    rmse_2 = tf.keras.metrics.RootMeanSquaredError()
    mape_2 = tf.keras.metrics.MeanAbsolutePercentageError()

    @tf.function
    def train_step_2(loss_inputs, inputs, targets, weights, option):
        with tf.GradientTape() as tape:
            pars = model_2(inputs)
            loss_value = rchi2_Loss(loss_inputs, pars, targets, weights, option)
        grads = tape.gradient(loss_value, model_2.trainable_weights)
        optimizer_2.apply_gradients(zip(grads, model_2.trainable_weights))
        return loss_value
        
    @tf.function
    def test_step_2(loss_inputs, inputs, targets, weights, option):
        pars = model_2(inputs)
        val_loss_value = rchi2_Loss(loss_inputs, pars, targets, weights, option)
        return val_loss_value
    
    # Functions to update the metrics
    # MAPE for a given parameter: accuracy = (100 - MAPE)
    @tf.function
    def metricWrapper_2(m, kin3_norm, pars_true):
        mape_2.reset_state()
        def mapeMetric_2():
            pars = model_2(kin3_norm)       
            mape_2.update_state(pars_true[:, m], pars[:, m])
            return tf.convert_to_tensor(mape_2.result(), np.float32)
        return mapeMetric_2()
    
    @tf.function
    def dvcs_metricWrapper_2(kin, kin3_norm):
        mape_2.reset_state()
        # get true dvcs
        dvcs_true = np.array([pseudo['dvcs']]).transpose() 
        # get DNN dvcs
        pars = model_2(kin3_norm)  
        k, QQ, x, t, phi = tf.split(kin, num_or_size_splits=5, axis=1)
        ReH, ReE, ReHt, dvcs_par = tf.split(pars, num_or_size_splits=4, axis=1) 
        ee, y, xi, Gamma, tmin, Ktilde_10, K = bkm10.SetKinematics(QQ, x, t, k)
        dvcs_dnn =  bkm10.DVCS_3CFFs(QQ, x, t, phi, ee, y, K, Gamma, ReH, ReE, ReHt, dvcs_par, 0)   # Pure DVCS cross-section
        def dvcs_mapeMetric_2():
            mape_2.update_state(dvcs_true, dvcs_dnn)
            return tf.convert_to_tensor(mape_2.result(), np.float32)
        return dvcs_mapeMetric_2()
    
    # F RMSE weighted over F_errors
    @tf.function
    def rmseMetric_2(kin, kin3_norm, pars_true, F_errors, option):    
        pars = model_2(kin3_norm)
        kin = tf.cast(kin, pars.dtype)        
        pars_true = tf.cast(pars_true, pars.dtype)
        F_dnn = tf.reshape(bkm10.total_xs(kin, pars, "t2", option), [-1])
        F_true = np.array([pseudo['F_true']]).transpose() 
        weights = 1. / F_errors
        rmse_2.update_state(F_true, F_dnn, sample_weight = weights)
        return tf.convert_to_tensor(rmse_2.result(), np.float32)
    

    # Keep results for plotting
    train_loss_results_3ParsDep = []
    val_loss_results_3ParsDep = []
    F_rmse_results_3ParsDep = []
    total_mape_results_3ParsDep = []
    ReH_mape_results_3ParsDep = []
    ReE_mape_results_3ParsDep = []
    ReHt_mape_results_3ParsDep = []
    dvcs_mape_results_3ParsDep = []
    predictions_results_3ParsDep = []

    patience = 1000
    wait = 0
    best = float("inf")

    option = "3CFFsDep"
    
    for epoch in range(epochs):
       
        loss_value_3ParsDep = train_step_2(kin_train, kin3_norm_train, F_train, Ferr_train, option)
        val_loss_value_3ParsDep = test_step_2(kin_test, kin3_norm_test, F_test, Ferr_test, option)

        # Update metrics    
        F_rmse_3ParsDep = rmseMetric_2(kin, kin3_norm, pars_true, pseudo['errF'], option)
        pars_mape_3ParsDep = [metricWrapper_2(m, kin3_norm, pars_true).numpy()  for m in range(3)]
        pars_mape_3ParsDep = np.append(pars_mape_3ParsDep, dvcs_metricWrapper_2(kin, kin3_norm))
        total_mape_3ParsDep = np.mean(pars_mape_3ParsDep) 
         
        # End epoch
        train_loss_results_3ParsDep.append(loss_value_3ParsDep)
        val_loss_results_3ParsDep.append(val_loss_value_3ParsDep)
        F_rmse_results_3ParsDep.append(F_rmse_3ParsDep)
        total_mape_results_3ParsDep.append(total_mape_3ParsDep)
        ReH_mape_results_3ParsDep.append(pars_mape_3ParsDep[0])
        ReE_mape_results_3ParsDep.append(pars_mape_3ParsDep[1])
        ReHt_mape_results_3ParsDep.append(pars_mape_3ParsDep[2])
        dvcs_mape_results_3ParsDep.append(pars_mape_3ParsDep[3])
        print("Epoch {:03d}: Loss: {:.3f} val_Loss: {:.3f} F_rmse: {:.5f} ReH_mape: {:.5f} ReE_mape: {:.5f} ReHt_mape: {:.5f} dvcs_mape: {:.5f} total_mape: {:.5f}".format(epoch, loss_value_3ParsDep, val_loss_value_3ParsDep, F_rmse_3ParsDep, pars_mape_3ParsDep[0], pars_mape_3ParsDep[1], pars_mape_3ParsDep[2], pars_mape_3ParsDep[3], total_mape_3ParsDep))
        # print("Epoch {:03d}: Loss: {:.3f} val_Loss: {:.3f} F_rmse: {:.5f} ReH_mape: {:.5f} ReE_mape: {:.5f} ReHt_mape: {:.5f} total_mape: {:.5f}".format(epoch, loss_value, val_loss_value, F_rmse, pars_mape[0], pars_mape[1], pars_mape[2], total_mape))

        # Reset training metrics at the end of each epoch
        rmse_2.reset_state()
        mape_2.reset_state()

        # Get prediction for one set (set 1) for visualization
        if epoch % 10 == 0:
            # predictions = model(kin3_norm[:1])
            predictions_3ParsDep = model_2.predict(kin3_norm[:1])
            predictions_results_3ParsDep.append(predictions_3ParsDep[:1])

        # Apply the early stopping strategy after 1000 epochs: stop the training if `total_mape` does not
        # decrease over a certain number of epochs.
        if early_stop:
            if epoch > 1000:
                wait += 1
                if total_mape_3ParsDep < best:
                    best = total_mape_3ParsDep
                    wait = 0
                if wait >= patience:
                    print(f'\nEarly stopping. No improvement in average MAPE in the last {patience} epochs.')
                    break
    # history = {'loss': train_loss_results, 'val_loss': val_loss_results, 'ReH_mape': ReH_mape_results, 'ReE_mape': ReE_mape_results, 'ReHt_mape': ReHt_mape_results, 'dvcs_mape': dvcs_mape_results, 'total_mape': total_mape_results}
    # tf.keras.models.save_model(model, 'models/'+GPD_MODEL+'/test/fit_replica_'+str(i)+'.keras') # need "tf.keras.models.save_model" to save custom layer
    # np.save('models/'+GPD_MODEL+'/test/history_fit_replica_'+str(i)+'.npy',history) 

    predictions_results_3ParsDep = np.array(predictions_results_3ParsDep)

    # -------------------------------------------------------------------------------------------------
    # Training with the 8 CFFs output parameters
    # ------------------------------------------------------------------------------------------------

    model_3 = build_model(8)
    model_3.summary()

    # Instantiate an optimizer to train the model.
    optimizer_3 = tf.keras.optimizers.Adam(learning_rate=0.00008)
    rmse_3 = tf.keras.metrics.RootMeanSquaredError()
    mape_3 = tf.keras.metrics.MeanAbsolutePercentageError()

    @tf.function
    def train_step_3(loss_inputs, inputs, targets, weights, option):
        with tf.GradientTape() as tape:
            pars = model_3(inputs)
            loss_value = rchi2_Loss(loss_inputs, pars, targets, weights, option)
        grads = tape.gradient(loss_value, model_3.trainable_weights)
        optimizer_3.apply_gradients(zip(grads, model_3.trainable_weights))
        return loss_value
        
    @tf.function
    def test_step_3(loss_inputs, inputs, targets, weights, option):
        pars = model_3(inputs)
        val_loss_value = rchi2_Loss(loss_inputs, pars, targets, weights, option)
        return val_loss_value
    
    # Functions to update the metrics
    # MAPE for a given parameter: accuracy = (100 - MAPE)
    @tf.function
    def metricWrapper_3(m, kin3_norm, pars_true):
        mape_3.reset_state()
        def mapeMetric_3():
            pars = model_3(kin3_norm)       
            mape_3.update_state(pars_true[:, m], pars[:, m])
            return tf.convert_to_tensor(mape_3.result(), np.float32)
        return mapeMetric_3()
    
    @tf.function
    def dvcs_metricWrapper_3(kin, kin3_norm):
        mape_3.reset_state()
        # get true dvcs
        dvcs_true = np.array([pseudo['dvcs']]).transpose() 
        # get DNN dvcs
        pars = model_3(kin3_norm)  
        k, QQ, x, t, phi = tf.split(kin, num_or_size_splits=5, axis=1)
        ReH, ReE, ReHt, ReEt, ImH, ImE, ImHt, ImEt = tf.split(pars, num_or_size_splits=8, axis=1) 
        ee, y, xi, Gamma, tmin, Ktilde_10, K = bkm10.SetKinematics(QQ, x, t, k)
        dvcs_dnn =  bkm10.DVCS(QQ, x, t, phi, ee, y, K, Gamma, ReH, ReE, ReHt, ReEt, ImH, ImE, ImHt, ImEt, 0)   # Pure DVCS cross-section
        def dvcs_mapeMetric_3():
            mape_3.update_state(dvcs_true, dvcs_dnn)
            return tf.convert_to_tensor(mape_3.result(), np.float32)
        return dvcs_mapeMetric_3()
    
    # F RMSE weighted over F_errors
    @tf.function
    def rmseMetric_3(kin, kin3_norm, F_errors, option):    
        pars = model_3(kin3_norm)
        kin = tf.cast(kin, pars.dtype)        
        F_dnn = tf.reshape(bkm10.total_xs(kin, pars, "t2", option), [-1])
        F_true = np.array([pseudo['F_true']]).transpose() 
        weights = 1. / F_errors
        rmse_3.update_state(F_true, F_dnn, sample_weight = weights)
        return tf.convert_to_tensor(rmse_3.result(), np.float32)
    

    # Keep results for plotting
    train_loss_results_allCFFs = []
    val_loss_results_allCFFs = []
    F_rmse_results_allCFFs = []
    total_mape_results_allCFFs = []
    ReH_mape_results_allCFFs = []
    ReE_mape_results_allCFFs = []
    ReHt_mape_results_allCFFs = []
    dvcs_mape_results_allCFFs = []
    predictions_results_allCFFs = []

    patience = 1000
    wait = 0
    best = float("inf")

    option = "all_CFFs"
    
    for epoch in range(epochs):
       
        loss_value_allCFFs = train_step_3(kin_train, kin3_norm_train, F_train, Ferr_train, option)
        val_loss_value_allCFFs = test_step_3(kin_test, kin3_norm_test, F_test, Ferr_test, option)

        # Update metrics    
        F_rmse_allCFFs = rmseMetric_3(kin, kin3_norm, pseudo['errF'], option)
        pars_mape_allCFFs = [metricWrapper_3(m, kin3_norm, pars_true).numpy()  for m in range(3)]
        pars_mape_allCFFs = np.append(pars_mape_allCFFs, dvcs_metricWrapper_3(kin, kin3_norm))
        total_mape_allCFFs = np.mean(pars_mape_allCFFs) 
         
        # End epoch
        train_loss_results_allCFFs.append(loss_value_allCFFs)
        val_loss_results_allCFFs.append(val_loss_value_allCFFs)
        F_rmse_results_allCFFs.append(F_rmse_allCFFs)
        total_mape_results_allCFFs.append(total_mape_allCFFs)
        ReH_mape_results_allCFFs.append(pars_mape_allCFFs[0])
        ReE_mape_results_allCFFs.append(pars_mape_allCFFs[1])
        ReHt_mape_results_allCFFs.append(pars_mape_allCFFs[2])
        dvcs_mape_results_allCFFs.append(pars_mape_allCFFs[3])
        print("Epoch {:03d}: Loss: {:.3f} val_Loss: {:.3f} F_rmse: {:.5f} ReH_mape: {:.5f} ReE_mape: {:.5f} ReHt_mape: {:.5f} dvcs_mape: {:.5f} total_mape: {:.5f}".format(epoch, loss_value_allCFFs, val_loss_value_allCFFs, F_rmse_allCFFs, pars_mape_allCFFs[0], pars_mape_allCFFs[1], pars_mape_allCFFs[2], pars_mape_allCFFs[3], total_mape_allCFFs))
        # print("Epoch {:03d}: Loss: {:.3f} val_Loss: {:.3f} F_rmse: {:.5f} ReH_mape: {:.5f} ReE_mape: {:.5f} ReHt_mape: {:.5f} total_mape: {:.5f}".format(epoch, loss_value, val_loss_value, F_rmse, pars_mape[0], pars_mape[1], pars_mape[2], total_mape))

        # Reset training metrics at the end of each epoch
        rmse_3.reset_state()
        mape_3.reset_state()

        # Get prediction for one set (set 1) for visualization
        if epoch % 10 == 0:
            # predictions = model(kin3_norm[:1])
            predictions_allCFFs = model_3.predict(kin3_norm[:1])
            predictions_results_allCFFs.append(predictions_allCFFs[:1])

        # Apply the early stopping strategy after 1000 epochs: stop the training if `total_mape` does not
        # decrease over a certain number of epochs.
        if early_stop:
            if epoch > 1000:
                wait += 1
                if total_mape_allCFFs < best:
                    best = total_mape_allCFFs
                    wait = 0
                if wait >= patience:
                    print(f'\nEarly stopping. No improvement in average MAPE in the last {patience} epochs.')
                    break
    # history = {'loss': train_loss_results, 'val_loss': val_loss_results, 'ReH_mape': ReH_mape_results, 'ReE_mape': ReE_mape_results, 'ReHt_mape': ReHt_mape_results, 'dvcs_mape': dvcs_mape_results, 'total_mape': total_mape_results}
    # tf.keras.models.save_model(model, 'models/'+GPD_MODEL+'/test/fit_replica_'+str(i)+'.keras') # need "tf.keras.models.save_model" to save custom layer
    # np.save('models/'+GPD_MODEL+'/test/history_fit_replica_'+str(i)+'.npy',history) 

    predictions_results_allCFFs = np.array(predictions_results_allCFFs)

    #--------------------------
    
    # Draw loss metrics and predition for only the first set for visualization as a function of the number of epochs.
    fig, axes = plt.subplots(3, sharex=True, figsize=(14, 10))
    fig.suptitle('Training Metrics')

    # loss vs epoch
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results, label = "train - dvcs const")
    axes[0].plot(val_loss_results, label = "val - dvcs const")
    axes[0].plot(train_loss_results_3ParsDep, label = "train - dvcs 3CFFs dep")
    axes[0].plot(val_loss_results_3ParsDep, label = "val - dvcs 3CFFs dep")
    axes[0].plot(train_loss_results_allCFFs, label = "train - dvcs all CFFs")
    axes[0].plot(val_loss_results_allCFFs, label = "val - dvcs all CFFs")
    axes[0].legend(title = 'Loss')

    axes[1].set_ylabel("F_RMSE", fontsize=14)
    axes[1].plot(F_rmse_results, label = "dvcs const")
    axes[1].plot(F_rmse_results_3ParsDep, label = "dvcs 3CFFs dep")
    axes[1].plot(F_rmse_results_allCFFs, label = "dvcs all CFFs")
    axes[1].legend(title = 'F_RMSE')

    axes[2].plot(total_mape_results, label = "dvcs const")
    axes[2].plot(total_mape_results_3ParsDep, label = "dvcs 3CFFs dep")
    axes[2].plot(total_mape_results_allCFFs, label = "dvcs all CFFs")
    axes[2].set_ylabel("Average Pars MAPE", fontsize=14)
    axes[2].set_xlabel("Epoch", fontsize=14)
    axes[2].legend(title = 'Average Pars MAPE')

    plt.savefig('./model_test/loss_metric.pdf')

    # Draw pars mape vs epoch
    fig, axs = plt.subplots(2, 2, figsize=(20, 15), sharey=False, tight_layout=True)
    axs[0,0].plot(ReH_mape_results, label = 'dvcs constant')
    axs[0,0].plot(ReH_mape_results_3ParsDep, label = 'dvcs 3CFFs dep')
    axs[0,0].plot(ReH_mape_results_allCFFs, label = 'dvcs all CFFs')
    axs[0,1].plot(ReE_mape_results, label = 'dvcs constant')
    axs[0,1].plot(ReE_mape_results_3ParsDep, label = 'dvcs 3CFFs dep')
    axs[0,1].plot(ReE_mape_results_allCFFs, label = 'dvcs all CFFs')
    axs[1,0].plot(ReHt_mape_results, label = 'dvcs constant')
    axs[1,0].plot(ReHt_mape_results_3ParsDep, label = 'dvcs 3CFFs dep')
    axs[1,0].plot(ReHt_mape_results_allCFFs, label = 'dvcs all CFFs')
    axs[1,1].plot(dvcs_mape_results, label = 'dvcs constant')
    axs[1,1].plot(dvcs_mape_results_3ParsDep, label = 'dvcs 3CFFs dep')
    axs[1,1].plot(dvcs_mape_results_allCFFs, label = 'dvcs all CFFs')
    axs[0,0].set_ylabel("$\mathfrak{Re}\mathcal{H}$_mape", fontsize = 18)
    axs[0,1].set_ylabel("$\mathfrak{Re}\mathcal{E}$_mape", fontsize = 18)
    axs[1,0].set_ylabel("$\mathfrak{Re}\mathcal{\widetilde{H}}$_mape", fontsize = 18)
    axs[1,1].set_ylabel("$dvcs$_mape", fontsize = 18)
    axs[0,0].legend(title = 'ReH MAPE')
    axs[0,1].legend(title = 'ReE MAPE')
    axs[1,0].legend(title = 'ReHt MAPE')
    axs[1,1].legend(title = 'DVCS MAPE')

    plt.savefig('./model_test/pars_mape.pdf')

    # Draw pars pred vs epoch
    fig2, axs2 = plt.subplots(2, 2, figsize=(20, 15), sharey=False, tight_layout=True)
    xepoch = range(0, epoch+1, 10) 
    axs2[0,0].plot(xepoch, predictions_results[:,:,0], label = 'dvcs constant')
    axs2[0,0].plot(xepoch, predictions_results_3ParsDep[:,:,0], label = 'dvcs 3CFFs dep')
    axs2[0,0].plot(xepoch, predictions_results_allCFFs[:,:,0], label = 'dvcs all CFFs')
    axs2[0,0].axhline(y = pseudo['ReH'][0], color = 'r', label = 'true = '+ str('%.3g' % pseudo['ReH'][0]))
    axs2[0,1].plot(xepoch, predictions_results[:,:,1], label = 'dvcs constant')
    axs2[0,1].plot(xepoch, predictions_results_3ParsDep[:,:,1], label = 'dvcs 3CFFs dep')
    axs2[0,1].plot(xepoch, predictions_results_allCFFs[:,:,1], label = 'dvcs all CFFs')
    axs2[0,1].axhline(y = pseudo['ReE'][0], color = 'r', label = 'true = '+ str('%.3g' % pseudo['ReE'][0]))
    axs2[1,0].plot(xepoch, predictions_results[:,:,2], label = 'dvcs constant')
    axs2[1,0].plot(xepoch, predictions_results_3ParsDep[:,:,2], label = 'dvcs 3CFFs dep')
    axs2[1,0].plot(xepoch, predictions_results_allCFFs[:,:,2], label = 'dvcs all CFFs')
    axs2[1,0].axhline(y = pseudo['ReHt'][0], color = 'r', label = 'true = '+ str('%.3g' % pseudo['ReHt'][0]))
    # dvcs
    ReH_3ParsDep = predictions_results_3ParsDep[:,:,0]
    ReE_3ParsDep = predictions_results_3ParsDep[:,:,1]
    ReHt_3ParsDep = predictions_results_3ParsDep[:,:,2]
    dvcs_par_3ParsDep = predictions_results_3ParsDep[:,:,3]

    ReH_allCFFs = predictions_results_allCFFs[:,:,0]
    ReE_allCFFs = predictions_results_allCFFs[:,:,1]
    ReHt_allCFFs = predictions_results_allCFFs[:,:,2]
    ReEt_allCFFs = predictions_results_allCFFs[:,:,3]
    ImH_allCFFs = predictions_results_allCFFs[:,:,4]
    ImE_allCFFs = predictions_results_allCFFs[:,:,5]
    ImHt_allCFFs = predictions_results_allCFFs[:,:,6]
    ImEt_allCFFs = predictions_results_allCFFs[:,:,7]

    ee, y, xi, Gamma, tmin, Ktilde_10, K = bkm10.SetKinematics(1.82, 0.343, -0.172, 5.75)
    dvcs_3ParsDep =  bkm10.DVCS_3CFFs(1.82, 0.343, -0.172, 0, ee, y, K, Gamma, ReH_3ParsDep, ReE_3ParsDep, ReHt_3ParsDep, dvcs_par_3ParsDep, 0)   # Pure DVCS cross-section
    dvcs_allCFFs =  bkm10.DVCS(1.82, 0.343, -0.172, 0, ee, y, K, Gamma, ReH_allCFFs, ReE_allCFFs, ReHt_allCFFs, ReEt_allCFFs, ImH_allCFFs, ImE_allCFFs, ImHt_allCFFs, ImEt_allCFFs, 0)   # Pure DVCS cross-section
        
    # axs2[1,1].plot(xepoch, predictions_results[:,:,3], label = 'prediction')
    axs2[1,1].plot(xepoch, predictions_results[:,:,3], label = 'dvcs constant')
    axs2[1,1].plot(xepoch, dvcs_3ParsDep, label = 'dvcs 3CFFs dep')
    axs2[1,1].plot(xepoch, dvcs_allCFFs, label = 'dvcs all CFFs')
    axs2[1,1].axhline(y = pseudo['dvcs'][0], color = 'r', label = 'true = '+ str('%.3g' % pseudo['dvcs'][0]))
    axs2[0,0].set_ylabel("$\mathfrak{Re}\mathcal{H}$", fontsize = 18)
    axs2[0,1].set_ylabel("$\mathfrak{Re}\mathcal{E}$", fontsize = 18)
    axs2[1,0].set_ylabel("$\mathfrak{Re}\mathcal{\widetilde{H}}$", fontsize = 18)
    axs2[1,1].set_ylabel("$dvcs$", fontsize = 18)
    axs2[0,0].legend(title = 'set 1')
    axs2[0,1].legend(title = 'set 1')
    axs2[1,0].legend(title = 'set 1')
    axs2[1,1].legend(title = 'set 1')

    plt.savefig('./model_test/results_set_1.pdf')
    plt.show()   

pseudo = get_data()  

# print(pseudo)

kin = np.dstack((pseudo['k'], pseudo['QQ'] , pseudo['xB'], pseudo['t'], pseudo['phi']))
kin = kin.reshape(kin.shape[1:]) # loss inputs

# print(kin)

for i in range(0, NUM_OF_REPLICAS):
    start = time.time()
    fit_replica(i, pseudo)
    print("Run Time: ", (time.time() - start)/60, "min")    
















