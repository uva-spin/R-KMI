import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold, train_test_split
from tensorflow_addons.activations import tanhshrink
from tensorflow_addons.optimizers import AdamW
sys.path.append('../')
from Formulation.BHDVCS_tf_modified import BHDVCStf
import matplotlib.pyplot as plt
import time

tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

bkm10 = BHDVCStf()

GPD_MODEL = 'basic'
NUM_OF_REPLICAS = 1
early_stop = False
replica = False
cross_validation = True

datafile = '/media/lily/Data/GPDs/DVCS/Pseudodata/JLabKinematics/withRealErrors/pseudo_'+GPD_MODEL+'_BKM10_Jlab_all_t2.csv'

epochs = 9000

# get (pseudo)data file
def get_data():
    df = pd.read_csv(datafile, dtype=np.float32)
    return df

# filtering the unique set values to prevent overfitting    
def filter_unique_sets(data):
    unique_sets = set()
    filtered_data = {key: [] for key in data.keys()}
    for i in range(len(data['set'])):
        if data['set'][i] not in unique_sets:
            unique_sets.add(data['set'][i])
            for key in data.keys():
    filtered_data = {key: np.array(value) for key, value in filtered_data.items()}
    return filtered_data

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

def build_model():
    # model 75, info path: /media/lily/Data/GPDs/ANN/KMI/tunning/kt_RS
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(40, activation="sigmoid", input_shape=(3,)), 
        tf.keras.layers.Dense(180, activation="tanhshrink"),
        tf.keras.layers.Dense(130, activation="tanh"),
        tf.keras.layers.Dense(20, activation="tanh"),
        tf.keras.layers.Dense(4, activation="linear") # ReH, ReE, ReHt, dvcs
    ])    
    return model

# Reduced chi2 custom Loss function (model predicted inside loss)
def rchi2_Loss(kin, pars, F_data, F_err):
    kin = tf.cast(kin, pars.dtype)
    F_dnn = tf.reshape(bkm10.total_xs(kin, pars), [-1])
    F_data = tf.cast(F_data, pars.dtype)
    F_err = tf.cast(F_err, pars.dtype)
    loss = tf.reduce_mean(tf.square( (F_dnn - F_data) / (F_err) ) )
    return loss

def fit_replica(i, pseudo):
    # ----- prepare input data -----------  
    pseduo = filter_unique_sets(pseudo)
    
    if replica:        
        data = gen_replica(pseudo) # generate replica
    else:
        data = pseudo  

    kin = np.dstack((data['k'], data['QQ'] , data['xB'], data['t'], data['phi']))
    kin = kin.reshape(kin.shape[1:]) # loss inputs
    QQ_norm, xB_norm, t_norm = normalize(data['QQ'] , data['xB'], data['t']) 
    kin3_norm = np.array([QQ_norm, xB_norm, t_norm]).transpose() # model inputs
    pars_true = np.array([pseudo['ReH'], pseudo['ReE'], pseudo['ReHtilde'], pseudo['dvcs']]).transpose() # true parameters
    # ---- split train and testing replica data samples ---- 
    if cross_validation:
        rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        for train_index, test_index in rkf.split(kin):
            kin_train, kin_test, kin3_norm_train, kin3_norm_test = kin[train_index], kin[test_index], kin3_norm[train_index], kin3_norm[test_index]
            F_train, F_test = data['F'][train_index], data['F'][test_index]
            Ferr_train, Ferr_test = data['errF'][train_index], data['errF'][test_index]
    else:
        kin_train, kin_test, kin3_norm_train, kin3_norm_test, F_train, F_test, Ferr_train, Ferr_test = train_test_split(kin, kin3_norm, data['F'], data['errF'], test_size=0.10, random_state=42)

    model = build_model()
    model.summary()
      
    # Instantiate an optimizer to train the model.
    optimizer = AdamW(learning_rate=0.0001, weight_decay=0.0001)
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mape = tf.keras.metrics.MeanAbsolutePercentageError()

    @tf.function
    def train_step(loss_inputs, inputs, targets, weights):
        with tf.GradientTape() as tape:
            pars = model(inputs)
            loss_value = rchi2_Loss(loss_inputs, pars, targets, weights)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value
        
    @tf.function
    def test_step(loss_inputs, inputs, targets, weights):
        pars = model(inputs)
        val_loss_value = rchi2_Loss(loss_inputs, pars, targets, weights)
        return val_loss_value
    
    # Functions to update the metrics
    # MAPE for a given parameter: accuracy = (100 - MAPE)
    @tf.function
    def metricWrapper(m, kin3_norm, pars_true):
        mape.reset_states()
        def mapeMetric():
            pars = model(kin3_norm)       
            mape.update_state(pars_true[:, m], pars[:, m])
            return tf.convert_to_tensor(mape.result(), np.float32)
        return mapeMetric()
    # F RMSE weighted over F_errors
    @tf.function
    def rmseMetric(kin, kin3_norm, pars_true, F_errors):    
        pars = model(kin3_norm)
        kin = tf.cast(kin, pars.dtype)        
        pars_true = tf.cast(pars_true, pars.dtype)
        F_dnn = tf.reshape(bkm10.total_xs(kin, pars), [-1])
        F_true = tf.reshape(bkm10.total_xs(kin, pars_true), [-1])
        weights = 1. / F_errors
        rmse.update_state(F_true, F_dnn, sample_weight = weights)
        return tf.convert_to_tensor(rmse.result(), np.float32)
 
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

    patience = 1000
    wait = 0
    best = float("inf")
    
    for epoch in range(epochs):
       
        loss_value = train_step(kin_train, kin3_norm_train, F_train, Ferr_train)
        val_loss_value = test_step(kin_test, kin3_norm_test, F_test, Ferr_test)

        # Update metrics    
        F_rmse = rmseMetric(kin, kin3_norm, pars_true, pseudo['errF'])
        pars_mape = [metricWrapper(m, kin3_norm, pars_true).numpy()  for m in range(4)]
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

        # Reset training metrics at the end of each epoch
        rmse.reset_states()
        mape.reset_states()

        # Get prediction for one set (set 1) for visualization
        if epoch % 10 == 0:
            # predictions = model(kin3_norm[:1])
            predictions = model.predict(kin3_norm[:1])
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
    history = {'loss': train_loss_results, 'val_loss': val_loss_results, 'ReH_mape': ReH_mape_results, 'ReE_mape': ReE_mape_results, 'ReHt_mape': ReHt_mape_results, 'dvcs_mape': dvcs_mape_results, 'total_mape': total_mape_results}
    tf.keras.models.save_model(model, 'models/'+GPD_MODEL+'/test/fit_replica_'+str(i)+'.keras') # need "tf.keras.models.save_model" to save custom layer
    np.save('models/'+GPD_MODEL+'/test/history_fit_replica_'+str(i)+'.npy',history) 

    predictions_results = np.array(predictions_results)
    
    # Draw loss metrics and predition for only the first set for visualization as a function of the number of epochs.
    fig, axes = plt.subplots(3, sharex=True, figsize=(14, 10))
    fig.suptitle('Training Metrics')

    # loss vs epoch
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)
    axes[0].plot(val_loss_results)
    axes[1].set_ylabel("F_RMSE", fontsize=14)
    axes[1].plot(F_rmse_results)
    axes[2].plot(total_mape_results)
    axes[2].set_ylabel("Average Pars MAPE", fontsize=14)
    axes[2].set_xlabel("Epoch", fontsize=14)

    # Draw pars mape vs epoch
    fig, axs = plt.subplots(2, 2, figsize=(20, 15), sharey=False, tight_layout=True)
    axs[0,0].plot(ReH_mape_results, label = 'mape')
    axs[0,1].plot(ReE_mape_results, label = 'mape')
    axs[1,0].plot(ReHt_mape_results, label = 'mape')
    axs[1,1].plot(dvcs_mape_results, label = 'mape')
    axs[0,0].legend(title = 'set 1')
    axs[0,0].set_ylabel("$\mathfrak{Re}\mathcal{H}$_mape", fontsize = 18)
    axs[0,1].set_ylabel("$\mathfrak{Re}\mathcal{E}$_mape", fontsize = 18)
    axs[1,0].set_ylabel("$\mathfrak{Re}\mathcal{\widetilde{H}}$_mape", fontsize = 18)
    axs[1,1].set_ylabel("$dvcs$_mape", fontsize = 18)

    # Draw pars pred vs epoch
    fig2, axs2 = plt.subplots(2, 2, figsize=(20, 15), sharey=False, tight_layout=True)
    xepoch = range(0, epoch+1, 10) 
    axs2[0,0].plot(xepoch, predictions_results[:,:,0], label = 'prediction')
    axs2[0,0].axhline(y = pseudo['ReH'][0], color = 'r', label = 'true = '+ str('%.3g' % pseudo['ReH'][0]))
    axs2[0,1].plot(xepoch, predictions_results[:,:,1], label = 'prediction')
    axs2[0,1].axhline(y = pseudo['ReE'][0], color = 'r', label = 'true = '+ str('%.3g' % pseudo['ReE'][0]))
    axs2[1,0].plot(xepoch, predictions_results[:,:,2], label = 'prediction')
    axs2[1,0].axhline(y = pseudo['ReHtilde'][0], color = 'r', label = 'true = '+ str('%.3g' % pseudo['ReHtilde'][0]))
    axs2[1,1].plot(xepoch, predictions_results[:,:,3], label = 'prediction')
    axs2[1,1].axhline(y = pseudo['dvcs'][0], color = 'r', label = 'true = '+ str('%.3g' % pseudo['dvcs'][0]))
    axs2[0,0].legend(title = 'set 1')
    axs2[0,0].set_ylabel("$\mathfrak{Re}\mathcal{H}$", fontsize = 18)
    axs2[0,1].set_ylabel("$\mathfrak{Re}\mathcal{E}$", fontsize = 18)
    axs2[1,0].set_ylabel("$\mathfrak{Re}\mathcal{\widetilde{H}}$", fontsize = 18)
    axs2[1,1].set_ylabel("$dvcs$", fontsize = 18)
    
    plt.show()   

pseudo = get_data()
pseduo = filter_unique_sets(pseudo)

print(pseudo)

kin = np.dstack((pseudo['k'], pseudo['QQ'] , pseudo['xB'], pseudo['t'], pseudo['phi']))
kin = kin.reshape(kin.shape[1:]) # loss inputs

print(kin)

for i in range(0, NUM_OF_REPLICAS):
    start = time.time()
    fit_replica(i, pseudo)
    print("Run Time: ", (time.time() - start)/60, "min")    
















