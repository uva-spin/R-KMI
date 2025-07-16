import sys
import ROOT
from ROOT import TGraph
import numpy as np
import tensorflow as tf
sys.path.append('../../../')
from Formulation.BHDVCS_tf_modified import TotalFLayer
from tensorflow_addons.activations import tanhshrink
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def load_local_model(i,set_num):
    model = tf.keras.models.load_model('models/'+GPD_MODEL+'/set_'+str(set_num)+'/fit_replica_'+str(i)+'.keras', custom_objects={'TotalFLayer': TotalFLayer, "tanhshrink": tanhshrink})
    # history = np.load('./models/'+GPD_MODEL+'/set_'+str(set_num)+'/history_fit_replica_'+str(i)+'.npy',allow_pickle='TRUE').item() # Load history
    return model

def predict_model(model, kin):
    model_Pars = np.array(tf.keras.backend.function(model.layers[0].input, model.layers[5].output)(kin))
    model_F = np.array(tf.keras.backend.function(model.layers[0].input, model.layers[7].output)(kin))

    model_dict = {'Pars': model_Pars, 'F': model_F}   
     
    return model_dict


NUM_OF_REPLICAS = 500
GPD_MODEL = 'basic'
NUM_OF_SETS = 5

myfile = ROOT.TFile( 'results.root', 'RECREATE' )

loss_dir = myfile.mkdir("loss")
replica_dir = myfile.mkdir("replicas")
xs_dir = myfile.mkdir("F")
loss_dir.cd()
create_folders("F")
create_folders("replicas")
create_folders("results")
# ROOT.gDirectory.WriteObject(hist.GetPtr(), "pT")
# myFile.Close()

# etadir.cd() and then etadir.WriteObject(...),

models = []

for i in range(0, NUM_OF_REPLICAS):
    imodel = load_model(i)
    models.append(imodel)

true = []
t = []
lmi = []
elmi = []
local = []
elocal = []

for set in range(1, NUM_OF_SETS+1):
    F = []
    pars = []
    F_local = []
    pars_local = []
    pseudo = get_np_set(set)
    kin = np.dstack((pseudo['k'], pseudo['QQ'],pseudo['xB'], pseudo['t'], pseudo['phi']))
    kin = kin.reshape(kin.shape[1:])    
    print("====> set  ", set)

    for i in range(0, NUM_OF_REPLICAS):    
        # print("---- replica: ", i)
        if ((i+1) % (NUM_OF_REPLICAS * 0.1)) == 0:
            print("currently at replica ",i+1,": ... ", (i+1)*100/NUM_OF_REPLICAS, "%")
        ipredictions = predict_model(models[i], kin)
        F_i = ipredictions['F']
        pars_i = ipredictions['Pars']
        F.append(F_i)
        pars.append(pars_i[0]) # just take one value since the parameters do not depend on phi
        # print("parameters lmi: ", pars_i[0])

        # Local model
        imodel_local = load_local_model(i, set)
        ipredictions_local = predict_model(imodel_local, kin)
        F_i_local = ipredictions_local['F']
        pars_i_local = ipredictions_local['Pars']
        F_local.append(F_i_local)
        pars_local.append(pars_i_local[0]) # just take one value since the parameters do not depend on phi
        # print("parameters local: ", pars_i_local[0])

    # LMI
    F = np.array(F)
    F_mean = np.mean(F, 0)
    F_sigma = np.std(F, 0)
    # local
    F_local = np.array(F_local)
    F_mean_local = np.mean(F_local, 0)
    F_sigma_local = np.std(F_local, 0)

    # LMI
    pars = np.array(pars)
    par = {'ReH': pars[:, 0], 'ReE': pars[:, 1], 'ReHt': pars[:, 2], 'dvcs': pars[:, 3]} 
    par_true = {'ReH': pseudo['ReH'][0], 'ReE': pseudo['ReE'][0], 'ReHt': pseudo['ReHtilde'][0], 'dvcs': pseudo['dvcs'][0]} 
    par_mean = {'ReH': np.mean(par['ReH'], 0), 'ReE': np.mean(par['ReE'], 0), 'ReHt': np.mean(par['ReHt'], 0), 'dvcs': np.mean(par['dvcs'], 0)} 
    par_sigma = {'ReH': np.std(par['ReH'], 0), 'ReE': np.std(par['ReE'], 0), 'ReHt': np.std(par['ReHt'], 0), 'dvcs': np.std(par['dvcs'], 0)} 
    # local
    pars_local = np.array(pars_local)
    par_local = {'ReH': pars_local[:, 0], 'ReE': pars_local[:, 1], 'ReHt': pars_local[:, 2], 'dvcs': pars_local[:, 3]} 
    par_mean_local = {'ReH': np.mean(par_local['ReH'], 0), 'ReE': np.mean(par_local['ReE'], 0), 'ReHt': np.mean(par_local['ReHt'], 0), 'dvcs': np.mean(par_local['dvcs'], 0)} 
    par_sigma_local = {'ReH': np.std(par_local['ReH'], 0), 'ReE': np.std(par_local['ReE'], 0), 'ReHt': np.std(par_local['ReHt'], 0), 'dvcs': np.std(par_local['dvcs'], 0)} 
    
    # Save set results
    pars_name = ['ReH', 'ReE', 'ReHt', 'dvcs']
    # kinematics
    t.append(pseudo['t'][0])
    true.append([par_true[name] for name in pars_name])
    lmi.append([par_mean[name] for name in pars_name])
    elmi.append([par_sigma[name] for name in pars_name])
    local.append([par_mean_local[name] for name in pars_name])
    elocal.append([par_sigma_local[name] for name in pars_name])

    # Draw xs graphs
    plt.errorbar(pseudo['phi'], pseudo['F'], yerr=pseudo['errF'], fmt="o", capsize=3, label ='basic', color='black')
    plt.plot(pseudo['phi'], F_mean, alpha=0.8, label ='LMI', color='red')
    plt.fill_between(pseudo['phi'], F_mean.flatten() - F_sigma.flatten(), F_mean.flatten() + F_sigma.flatten(), color='red', alpha=0.3)
    plt.plot(pseudo['phi'], F_mean_local, alpha=0.8, label ='local', color='black')
    plt.fill_between(pseudo['phi'], F_mean_local.flatten() - F_sigma_local.flatten(), F_mean_local.flatten() + F_sigma_local.flatten(), color='black', alpha=0.3)
    plt.xlabel("$\phi[deg]$", fontsize = 16)
    plt.ylabel("$d^{4}\sigma_{UU}[nb/GeV^4]$", fontsize = 16)
    plt.legend(fontsize = 16, loc ="best")
    plt.savefig('F/' + 'F_set_' + str(set) + '.png', dpi = 300)
    plt.close()

    # Draw replica distributions
    fig, axs = plt.subplots(2, 2, figsize=(20, 15), sharey=False, tight_layout=True)
    bins_ReH = np.histogram(np.hstack((par['ReH'], par_local['ReH'])), bins=20)[1] #get the bin edges
    bins_ReE = np.histogram(np.hstack((par['ReE'], par_local['ReE'])), bins=20)[1] #get the bin edges
    bins_ReHt = np.histogram(np.hstack((par['ReHt'], par_local['ReHt'])), bins=20)[1] #get the bin edges
    bins_dvcs = np.histogram(np.hstack((par['dvcs'], par_local['dvcs'])), bins=80)[1] #get the bin edges
    axs[0,0].hist(par['ReH'],color='red', alpha=0.3, bins = bins_ReH)
    axs[0,1].hist(par['ReE'],color='red', alpha=0.3, bins = bins_ReE)
    axs[1,0].hist(par['ReHt'],color='red', alpha=0.3, bins = bins_ReHt)
    axs[1,1].hist(par['dvcs'],color='red', alpha=0.3, bins = bins_dvcs)
    axs[0,0].hist(par_local['ReH'],color='black', alpha=0.3, bins = bins_ReH)
    axs[0,1].hist(par_local['ReE'],color='black', alpha=0.3, bins = bins_ReE)
    axs[1,0].hist(par_local['ReHt'],color='black', alpha=0.3, bins = bins_ReHt)
    axs[1,1].hist(par_local['dvcs'],color='black', alpha=0.3, bins = bins_dvcs)

    ln_true = []
    ln_true.append(axs[0,0].axvline(x = par_true['ReH'], color = 'b', label = 'true = '+ str('%.3g' % par_true['ReH'])))
    ln_true.append(axs[0,1].axvline(x = par_true['ReE'], color = 'b', label = 'true = '+ str('%.3g' % par_true['ReE'])))
    ln_true.append(axs[1,0].axvline(x = par_true['ReHt'], color = 'b', label = 'true = '+ str('%.3g' % par_true['ReHt'])))
    ln_true.append(axs[1,1].axvline(x = par_true['dvcs'], color = 'b', label = 'true = '+ str('%.3g' % par_true['dvcs'])))
    ln_mean_lmi = []
    ln_mean_lmi.append(axs[0,0].axvline(x = par_mean['ReH'], color = 'r', label = '$mean_{LMI}$ = '+ str('%.3g' % par_mean['ReH'])))      
    ln_mean_lmi.append(axs[0,1].axvline(x = par_mean['ReE'], color = 'r', label = '$mean_{LMI}$ = '+ str('%.3g' % par_mean['ReE'])))   
    ln_mean_lmi.append(axs[1,0].axvline(x = par_mean['ReHt'], color = 'r', label = '$mean_{LMI}$ = '+ str('%.3g' % par_mean['ReHt'])))      
    ln_mean_lmi.append(axs[1,1].axvline(x = par_mean['dvcs'], color = 'r', label = '$mean_{LMI}$ = '+ str('%.3g' % par_mean['dvcs'])))
    ln_mean_loc = []
    ln_mean_loc.append(axs[0,0].axvline(x = par_mean_local['ReH'], color = 'black', label = '$mean_{Loc}$ = '+ str('%.3g' % par_mean_local['ReH'])))        
    ln_mean_loc.append(axs[0,1].axvline(x = par_mean_local['ReE'], color = 'black', label = '$mean_{Loc}$ = '+ str('%.3g' % par_mean_local['ReE'])))
    ln_mean_loc.append(axs[1,0].axvline(x = par_mean_local['ReHt'], color = 'black', label = '$mean_{Loc}$ = '+ str('%.3g' % par_mean_local['ReHt'])))    
    ln_mean_loc.append(axs[1,1].axvline(x = par_mean_local['dvcs'], color = 'black', label = '$mean_{Loc}$ = '+ str('%.3g' % par_mean_local['dvcs'])))
    ln_std_lmi = []
    ln_std_lmi.append('$\sigma_{LMI}$ = '+ str('%.3g' % par_sigma['ReH']))
    ln_std_lmi.append('$\sigma_{LMI}$ = '+ str('%.3g' % par_sigma['ReE']))
    ln_std_lmi.append('$\sigma_{LMI}$ = '+ str('%.3g' % par_sigma['ReHt']))
    ln_std_lmi.append('$\sigma_{LMI}$ = '+ str('%.2g' % par_sigma['dvcs']))
    ln_std_loc = []
    ln_std_loc.append('$\sigma_{Loc}$ = '+ str('%.3g' % par_sigma_local['ReH']))
    ln_std_loc.append('$\sigma_{Loc}$ = '+ str('%.3g' % par_sigma_local['ReE']))
    ln_std_loc.append('$\sigma_{Loc}$ = '+ str('%.3g' % par_sigma_local['ReHt']))
    ln_std_loc.append('$\sigma_{Loc}$ = '+ str('%.2g' % par_sigma_local['dvcs']))

    lmi_tx = 'LMI$(\leq 5)$'
    loc_tx = 'Local'
    lmi_box = mpatches.Patch(color='red', alpha=0.3)
    loc_box = mpatches.Patch(color='black', alpha=0.3)
    extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    axs[0,0].set_xlabel("$\mathfrak{Re}\mathcal{H}$", fontsize = 18)
    axs[0,0].set_ylabel("$frequency$", fontsize = 18)
    axs[0,0].legend([ln_true[0], lmi_box, ln_mean_lmi[0], extra, loc_box, ln_mean_loc[0], extra],[ln_true[0].get_label(), lmi_tx, ln_mean_lmi[0].get_label(), ln_std_lmi[0], loc_tx, ln_mean_loc[0].get_label(), ln_std_loc[0]], fontsize = 18, loc ="best")
    axs[0,1].set_xlabel("$\mathfrak{Re}\mathcal{E}$", fontsize = 18)
    axs[0,1].set_ylabel("$frequency$", fontsize = 18)
    axs[0,1].legend([ln_true[1], lmi_box, ln_mean_lmi[1], extra, loc_box, ln_mean_loc[1], extra],[ln_true[1].get_label(), lmi_tx, ln_mean_lmi[1].get_label(), ln_std_lmi[1], loc_tx, ln_mean_loc[1].get_label(), ln_std_loc[1]], fontsize = 18, loc ="best")
    axs[1,0].set_xlabel("$\mathfrak{Re}\mathcal{\widetilde{H}}$", fontsize = 18)
    axs[1,0].set_ylabel("$frequency$", fontsize = 18)
    axs[1,0].legend([ln_true[2], lmi_box, ln_mean_lmi[2], extra, loc_box, ln_mean_loc[2], extra],[ln_true[2].get_label(), lmi_tx, ln_mean_lmi[2].get_label(), ln_std_lmi[2], loc_tx, ln_mean_loc[2].get_label(), ln_std_loc[2]], fontsize = 18, loc ="best")
    axs[1,1].set_xlabel("$dvcs$", fontsize = 18)
    axs[1,1].set_ylabel("$frequency$", fontsize = 18)
    # axs[1,1].set_xlim(left=0)
    axs[1,1].set_xlim(-0.02, 0.06)
    axs[1,1].legend([ln_true[3], lmi_box, ln_mean_lmi[3], extra, loc_box, ln_mean_loc[3], extra],[ln_true[3].get_label(), lmi_tx, ln_mean_lmi[3].get_label(), ln_std_lmi[3], loc_tx, ln_mean_loc[3].get_label(), ln_std_loc[3]], fontsize = 18, loc ="best")
    fig.subplots_adjust(top=0.88)
    fig.suptitle('set_' + str(set), fontsize = 18)       
    plt.savefig('replicas/' + 'replicas_set_' + str(set) + '.png', dpi = 300)
    plt.close()


true = np.array(true)
t = np.array(t)
lmi = np.array(lmi)
elmi = np.array(elmi)
local = np.array(local)
elocal = np.array(elocal)

# CFFs vs t graphs
fig, axs = plt.subplots(2, 2, figsize=(20, 15), sharey=False, tight_layout=True)
axs[0,0].errorbar(-t, true[:, 0], xerr=0.008,  ls = "None", capsize=0, label ='true', color='blue', elinewidth=5, alpha = 0.5)
axs[0,0].errorbar(-t-0.001, lmi[:, 0], yerr=elmi[:, 0],  fmt = 'o', capsize=3, label ='LMI$(\leq 5)$', color='red', alpha = 0.8)
axs[0,0].errorbar(-t+0.001, local[:, 0], yerr=elocal[:, 0],  fmt = 'o', capsize=3, label ='Local', color='black', alpha = 0.8)
axs[0,1].errorbar(-t, true[:, 1], xerr=0.008,  ls = "None", capsize=0, label ='true', color='blue', elinewidth=5, alpha = 0.5)
axs[0,1].errorbar(-t-0.001, lmi[:, 1], yerr=elmi[:, 1],  fmt = 'o', capsize=3, label ='LMI$(\leq 5)$', color='red', alpha = 0.8)
axs[0,1].errorbar(-t+0.001, local[:, 1], yerr=elocal[:, 1],  fmt = 'o', capsize=3, label ='Local', color='black', alpha = 0.8)
axs[1,0].errorbar(-t, true[:, 2], xerr=0.008,  ls = "None", capsize=0, label ='true', color='blue', elinewidth=5, alpha = 0.5)
axs[1,0].errorbar(-t-0.001, lmi[:, 2], yerr=elmi[:, 2],  fmt = 'o', capsize=3, label ='LMI$(\leq 5)$', color='red', alpha = 0.8)
axs[1,0].errorbar(-t+0.001, local[:, 2], yerr=elocal[:, 2],  fmt = 'o', capsize=3, label ='Local', color='black', alpha = 0.8)
axs[1,1].errorbar(-t, true[:, 3], xerr=0.008,  ls = "None", capsize=0, label ='true', color='blue', elinewidth=5, alpha = 0.5)
axs[1,1].errorbar(-t-0.001, lmi[:, 3], yerr=elmi[:, 3],  fmt = 'o', capsize=3, label ='LMI$(\leq 5)$', color='red', alpha = 0.8)
axs[1,1].errorbar(-t+0.001, local[:, 3], yerr=elocal[:, 3],  fmt = 'o', capsize=3, label ='Local', color='black', alpha = 0.8)

axs[0,0].set_ylabel("$\mathfrak{Re}\mathcal{H}$", fontsize = 18)
axs[0,0].set_xlabel("$t[GeV^2]$", fontsize = 18)
axs[0,0].legend(fontsize = 18, loc ="best")
axs[0,1].set_ylabel("$\mathfrak{Re}\mathcal{E}$", fontsize = 18)
axs[0,1].set_xlabel("$t[GeV^2]$", fontsize = 18)
axs[0,1].legend(fontsize = 18, loc ="best")
axs[1,0].set_ylabel("$\mathfrak{Re}\mathcal{\widetilde{H}}$", fontsize = 18)
axs[1,0].set_xlabel("$t[GeV^2]$", fontsize = 18)
axs[1,0].legend(fontsize = 18, loc ="best")
axs[1,1].set_ylabel("$dvcs$", fontsize = 18)
axs[1,1].set_xlabel("$t[GeV^2]$", fontsize = 18)
axs[1,1].legend(fontsize = 18, loc ="best")
fig.subplots_adjust(top=0.88)
fig.suptitle('basic model - sets 1 to 5', fontsize = 18)  

plt.savefig('results/cffs_vs_t_sets_1to5.png', dpi = 300)
# plt.show()
plt.close()

myfile.Close()
    
