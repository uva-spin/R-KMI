import sys
import ROOT
from ROOT import TGraph
import numpy as np
import tensorflow as tf
sys.path.append('../../../')
from Formulation.BHDVCS_tf_modified import TotalFLayer
from tensorflow_addons.activations import tanhshrink
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from array import array
import os

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

def get_np_set(set):
    set_cut = "set == " + str(set)
    df = ROOT.RDF.FromCSV('/media/lily/Data/GPDs/DVCS/Pseudodata/JLabKinematics/withRealErrors/pseudo_'+GPD_MODEL+'_BKM10_Jlab_all_t2.csv')
    npset = df.Filter(set_cut).AsNumpy() 
    return npset

def load_model(i):
    model = tf.keras.models.load_model('models/'+GPD_MODEL+'/fit_replica_'+str(i)+'.keras', custom_objects={'TotalFLayer': TotalFLayer, "tanhshrink": tanhshrink})
    history = np.load('./models/'+GPD_MODEL+'/history_fit_replica_'+str(i)+'.npy',allow_pickle='TRUE').item() # Load history
    nepochs = len(history['loss'])
    epochs = np.linspace(1, nepochs, nepochs)
    train_loss = np.array(history['loss'])
    val_loss = np.array(history['val_loss'])

    gr = TGraph( nepochs, epochs,  train_loss )
    gr.SetLineColor( 4 )
    gr.SetLineWidth( 4 )
    gr.SetName( 'train_loss' )
        
    gr1 = TGraph( nepochs, epochs,  val_loss )
    gr1.SetLineColor( 2 )
    gr1.SetLineWidth( 4 )
    gr1.SetName( 'validation_loss' )
 
    str_obj = str(i)
    name = 'loss_replica_'+str_obj
    mg = ROOT.TMultiGraph(name,"Loss Curve")
    mg.Add(gr, 'AL')
    mg.Add(gr1, 'AL')
    mg.GetXaxis().SetTitle( 'epoch' )
    mg.GetYaxis().SetTitle( 'loss' )
    # mg.Draw( 'AL' )
    mg.Write() 
   
    return model

def predict_model(model, kin):
    model_Pars = np.array(tf.keras.backend.function(model.layers[0].input, model.layers[5].output)(kin))
    model_F = np.array(tf.keras.backend.function(model.layers[0].input, model.layers[7].output)(kin))

    model_dict = {'Pars': model_Pars, 'F': model_F}   
     
    return model_dict

NUM_OF_REPLICAS = 5
GPD_MODEL = 'basic'
NUM_OF_SETS = 1

myfile = ROOT.TFile( 'results.root', 'RECREATE' )

loss_dir = myfile.mkdir("loss")
replica_dir = myfile.mkdir("replicas")
xs_dir = myfile.mkdir("F")
loss_dir.cd()
create_folders("F")
create_folders("replicas")
# ROOT.gDirectory.WriteObject(hist.GetPtr(), "pT")
# myFile.Close()

# etadir.cd() and then etadir.WriteObject(...),

models = []

for i in range(0, NUM_OF_REPLICAS):
    imodel = load_model(i)
    models.append(imodel)

for set in range(1, NUM_OF_SETS+1):
    F = []
    pars = []
    pseudo = get_np_set(set)
    kin = np.dstack((pseudo['k'], pseudo['QQ'],pseudo['xB'], pseudo['t'], pseudo['phi']))
    kin = kin.reshape(kin.shape[1:])    
    print("====> set  ", set)

    for i in range(0, NUM_OF_REPLICAS):    
        print("---- replica: ", i)
        # print("kinematics: ", kin)
        ipredictions = predict_model(models[i], kin)
        F_i = ipredictions['F']
        pars_i = ipredictions['Pars']
        F.append(F_i)
        pars.append(pars_i[0]) # just take one value since the parameters do not depend on phi
        print("parameters: ", pars_i[0])

    F = np.array(F)
    F_mean = np.mean(F, 0)
    F_sigma = np.std(F, 0)

    pars = np.array(pars)
    par = {'ReH': pars[:, 0], 'ReE': pars[:, 1], 'ReHt': pars[:, 2], 'dvcs': pars[:, 3]} 
    par_true = {'ReH': pseudo['ReH'][0], 'ReE': pseudo['ReE'][0], 'ReHt': pseudo['ReHtilde'][0], 'dvcs': pseudo['dvcs'][0]} 
    par_mean = {'ReH': np.mean(par['ReH'], 0), 'ReE': np.mean(par['ReE'], 0), 'ReHt': np.mean(par['ReHt'], 0), 'dvcs': np.mean(par['dvcs'], 0)} 
    par_sigma = {'ReH': np.std(par['ReH'], 0), 'ReE': np.std(par['ReE'], 0), 'ReHt': np.std(par['ReHt'], 0), 'dvcs': np.std(par['dvcs'], 0)} 

    
    # Draw xs graphs
    plt.errorbar(pseudo['phi'], pseudo['F'], yerr=pseudo['errF'], fmt="o", capsize=3, label ='basic', color='black')
    plt.plot(pseudo['phi'], F_mean, alpha=0.5, label ='LMI', color='red')
    plt.fill_between(pseudo['phi'], F_mean.flatten() - F_sigma.flatten(), F_mean.flatten() + F_sigma.flatten(), color='red', alpha=0.3)
    plt.xlabel("$\phi[deg]$", fontsize = 18)
    plt.ylabel("$d^{4}\sigma_{UU}[nb/GeV^4]$", fontsize = 18)
    plt.legend(fontsize = 18, loc ="upper left")
    plt.savefig('F/' + 'F_set_' + str(set) + '.png', dpi = 300)
    plt.close()

    # Draw replica distributions
    fig, axs = plt.subplots(2, 2, figsize=(20, 15), sharey=False, tight_layout=True)
    # We can set the number of bins with the *bins* keyword argument.
    axs[0,0].hist(par['ReH'],color='red', alpha=0.3)
    axs[0,1].hist(par['ReE'],color='red', alpha=0.3)
    axs[1,0].hist(par['ReHt'],color='red', alpha=0.3)
    axs[1,1].hist(par['dvcs'],color='red', alpha=0.3)

    lReH_true = axs[0,0].axvline(x = par_true['ReH'], color = 'b', label = 'true = '+ str('%.3g' % par_true['ReH']))
    lReH_mean = axs[0,0].axvline(x = par_mean['ReH'], color = 'r', label = 'mean = '+ str('%.3g' % par_mean['ReH']))        
    axs[0,1].axvline(x = par_true['ReE'], color = 'b', label = 'true = '+ str('%.3g' % par_true['ReE']))
    axs[0,1].axvline(x = par_mean['ReE'], color = 'r', label = 'mean = '+ str('%.3g' % par_mean['ReE']))
    axs[1,0].axvline(x = par_true['ReHt'], color = 'b', label = 'true = '+ str('%.3g' % par_true['ReHt']))
    axs[1,0].axvline(x = par_mean['ReHt'], color = 'r', label = 'mean = '+ str('%.3g' % par_mean['ReHt']))    
    axs[1,1].axvline(x = par_true['dvcs'], color = 'b', label = 'true = '+ str('%.3g' % par_true['dvcs']))
    axs[1,1].axvline(x = par_mean['dvcs'], color = 'r', label = 'mean = '+ str('%.3g' % par_mean['dvcs']))

    tx_ReH_sigma = 'sigma = '+ str('%.3g' % par_sigma['ReH'])
    # text2 = 'modeled ratign curve $Q = 2.71H^2 - 2.20H + 0.98$'
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    # plt.legend([p.get_label(), tx_ReH_sigma, text2], loc='upper left', title='Legend')

    axs[0,0].set_xlabel("$\mathfrak{Re}\mathcal{H}$", fontsize = 18)
    axs[0,0].set_ylabel("$frequency$", fontsize = 18)
    axs[0,0].legend([lReH_true, lReH_mean, extra],[lReH_true.get_label(), lReH_mean.get_label(), tx_ReH_sigma], fontsize = 18, loc ="best")
    axs[0,1].set_xlabel("$\mathfrak{Re}\mathcal{E}$", fontsize = 18)
    axs[0,1].set_ylabel("$frequency$", fontsize = 18)
    axs[0,1].legend(fontsize = 18, loc ="best")
    axs[1,0].set_xlabel("$\mathfrak{Re}\mathcal{\widetilde{H}}$", fontsize = 18)
    axs[1,0].set_ylabel("$frequency$", fontsize = 18)
    axs[1,0].legend(fontsize = 18, loc ="best")
    axs[1,1].set_xlabel("$dvcs$", fontsize = 18)
    axs[1,1].set_ylabel("$frequency$", fontsize = 18)
    axs[1,1].legend(fontsize = 18, loc ="best")
    fig.subplots_adjust(top=0.88)
    fig.suptitle('set_' + str(set), fontsize = 18)       
    plt.savefig('replicas/' + 'replicas_set_' + str(set) + '.png', dpi = 300)
    plt.close()

# pseudo = get_np_set(3) 

# kin = np.dstack((pseudo['k'], pseudo['QQ'],pseudo['xB'], pseudo['t'], pseudo['phi']))
# kin = kin.reshape(kin.shape[1:])

# plt.errorbar(pseudo['phi'], pseudo['F'], yerr=pseudo['errF'], fmt="o", capsize=3, label ='basic', color='black')

# F = []
# pars = []

# for i in range(0, NUM_OF_REPLICAS):
#     imodel = load_model(i)
#     ipredictions = predict_model(imodel, kin)
#     F_i = ipredictions['F']
#     pars_i = ipredictions['Pars']
#     F.append(F_i)
#     pars.append(pars_i[0]) # just take one value since the parameters do not depend on phi
    
# F = np.array(F)
# F_mean = np.mean(F, 0)
# F_sigma = np.std(F, 0)

# pars = np.array(pars)
# par = {'ReH': pars[:, 0], 'ReE': pars[:, 1], 'ReHt': pars[:, 2], 'dvcs': pars[:, 3]} 
# par_true = {'ReH': pseudo['ReH'][0], 'ReE': pseudo['ReE'][0], 'ReHt': pseudo['ReHtilde'][0], 'dvcs': pseudo['dvcs'][0]} 
# par_mean = {'ReH': np.mean(par['ReH'], 0), 'ReE': np.mean(par['ReE'], 0), 'ReHt': np.mean(par['ReHt'], 0), 'dvcs': np.mean(par['dvcs'], 0)} 

# plt.figure(1)
# plt.plot(pseudo['phi'], F_mean, alpha=0.5, label ='LMI', color='red')
# plt.fill_between(pseudo['phi'], F_mean.flatten() - F_sigma.flatten(), F_mean.flatten() + F_sigma.flatten(), color='red', alpha=0.3)
# plt.xlabel("$\phi[deg]$", fontsize = 18)
# plt.ylabel("$d^{4}\sigma_{UU}[nb/GeV^4]$", fontsize = 18)
# plt.legend(fontsize = 18, loc ="upper left")

# fig, axs = plt.subplots(2, 2, sharey=False, tight_layout=True)

# # We can set the number of bins with the *bins* keyword argument.
# axs[0,0].hist(par['ReH'],color='red', alpha=0.3)
# axs[0,1].hist(par['ReE'],color='red', alpha=0.3)
# axs[1,0].hist(par['ReHt'],color='red', alpha=0.3)
# axs[1,1].hist(par['dvcs'],color='red', alpha=0.3)

# axs[0,0].axvline(x = par_mean['ReH'], color = 'r', label = 'mean')
# axs[0,0].axvline(x = par_true['ReH'], color = 'b', label = 'true')
# axs[0,1].axvline(x = par_mean['ReE'], color = 'r', label = 'mean')
# axs[0,1].axvline(x = par_true['ReE'], color = 'b', label = 'true')
# axs[1,0].axvline(x = par_mean['ReHt'], color = 'r', label = 'mean')
# axs[1,0].axvline(x = par_true['ReHt'], color = 'b', label = 'true')
# axs[1,1].axvline(x = par_mean['dvcs'], color = 'r', label = 'mean')
# axs[1,1].axvline(x = par_true['dvcs'], color = 'b', label = 'true')

# axs[0,0].set_xlabel("$\mathfrak{Re}\mathcal{H}$", fontsize = 18)
# axs[0,0].set_ylabel("$frequency$", fontsize = 18)
# axs[0,0].legend(fontsize = 18, loc ="upper left")
# axs[0,1].set_xlabel("$\mathfrak{Re}\mathcal{E}$", fontsize = 18)
# axs[0,1].set_ylabel("$frequency$", fontsize = 18)
# axs[0,1].legend(fontsize = 18, loc ="upper left")
# axs[1,0].set_xlabel("$\mathfrak{Re}\mathcal{\widetilde{H}}$", fontsize = 18)
# axs[1,0].set_ylabel("$frequency$", fontsize = 18)
# axs[1,0].legend(fontsize = 18, loc ="upper left")
# axs[1,1].set_xlabel("$dvcs$", fontsize = 18)
# axs[1,1].set_ylabel("$frequency$", fontsize = 18)
# axs[1,1].legend(fontsize = 18, loc ="upper left")

# plt.show()  

myfile.Close()
    
