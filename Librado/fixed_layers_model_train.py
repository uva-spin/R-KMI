import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import ast

from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow_addons.activations import tanhshrink
from tensorflow_addons.optimizers import AdamW
print(sys.path)
sys.path.append('/sfs/qumulo/qhome/lba9wf/R-KMI')
print(sys.path)
from Formulation.BHDVCS_tf_modified import BHDVCStf
import matplotlib.pyplot as plt
import time

tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

bkm10 = BHDVCStf()

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Train the BHDVCS model with given parameters.')
parser.add_argument('--activation', type=str, help='Activation function list', required=True)
parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer', required=True)
parser.add_argument('--batch_size', type=int, help='Batch size for training', required=True)
parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)
parser.add_argument('--optimizer', type=str, help='Optimizer', required=True)
parser.add_argument('--output_dir', type=str, help='Output directory for saving models and history', required=True)
parser.add_argument('--nodes_per_layer', type=str, help='List of nodes per layer')

mape_reh = []
mape_ree = []
mape_rehtilde = []
mape_dvcs = []
mape_total = []

args = parser.parse_args()

tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

bkm10 = BHDVCStf()
GPD_MODEL = 'basic'
NUM_OF_REPLICAS = args.batch_size
early_stop = False
replica = True
cross_validation = True

datafile = '/sfs/qumulo/qhome/lba9wf/R-KMI/Pseudodata/pseudo_basic_BKM10_Jlab_all_t2.csv'
epochs = args.epochs

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

'''def build_model():
    model = tf.keras.Sequential()
    tf.keras.layers.Dense(40, activation="sigmoid", input_shape=(3,)),
    for _ in range(args.layers - 1):  # Assuming one input layer and args.layers-1 hidden layers
        tf.keras.layers.Dense(nodes, activation=activation_function),
    model.add(tf.keras.layers.Dense(4, activation="linear"))  # Output layer
    return model'''
    
'''def build_model():
    model = tf.keras.Sequential
    nodes_per_layer = ast.literal_eval(args.nodes_per_layer)
    model.add(tf.keras.layers.Dense(nodes_per_layer[0], activation=args.activation, input_shape=(3,)))
    for nodes in nodes_per_layer[1:]:
        model.add(tf.keras.layers.Dense(nodes, activation=args.activation))
    # Add the output layer
    model.add(tf.keras.layers.Dense(4, activation='linear'))
    return model'''
    
def build_model():
    model = tf.keras.Sequential()
    nodes_per_layer = ast.literal_eval(args.nodes_per_layer)
    print(f"nodes_per_layer: ", nodes_per_layer)
    activation_functions = ast.literal_eval(args.activation)
    print(f"activation: ", activation_functions)
    
    if isinstance(nodes_per_layer, list) and len(nodes_per_layer) > 0 and isinstance(activation_functions, list) and len(activation_functions) == len(nodes_per_layer):
        model.add(tf.keras.layers.Dense(nodes_per_layer[0], activation=activation_functions[0], input_shape=(3,)))
        
        for nodes, activation in zip(nodes_per_layer[1:], activation_functions[1:]):
            model.add(tf.keras.layers.Dense(nodes, activation=activation))
    else:
        raise ValueError("nodes_per_layer and activation_function must be lists of the same length and non-empty")
    
    model.add(tf.keras.layers.Dense(4, activation='linear'))
    
    return model
    
'''def build_model():
    # model 75, info path: /media/lily/Data/GPDs/ANN/KMI/tunning/kt_RS
    for _ in range(args.layers - 1):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(40, activation="sigmoid", input_shape=(3,)), 
            tf.keras.layers.Dense(180, activation="tanhshrink"),
            tf.keras.layers.Dense(130, activation="tanh"),
            tf.keras.layers.Dense(20, activation="tanh"),
            tf.keras.layers.Dense(4, activation="linear") # ReH, ReE, ReHt, dvcs
        ])    
    return model'''

'''def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(layers_info[0][0], activation=layers_info[0][1], input_shape=input_shape))
    # Add subsequent layers
    for nodes, activation in layers_info[1:]:
        model.add(tf.keras.layers.Dense(nodes, activation=activation))
    # Add the output layer
    model.add(tf.keras.layers.Dense(output_size, activation='linear'))
    return model'''

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
    
    for epoch in range(args.epochs):
       
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
        mape_reh.append(pars_mape[0])
        mape_ree.append(pars_mape[1])
        mape_rehtilde.append(pars_mape[2])
        mape_dvcs.append(pars_mape[3])
        mape_total.append(total_mape)
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
    print(sys.path)
    average_mape = np.mean(total_mape_results)
    os.makedirs(f'{args.output_dir}/models', exist_ok=True)
    print(sys.path)
    tf.keras.models.save_model(model, 'models/fit_replica_'+str(i)+'.keras') # need "tf.keras.models.save_model" to save custom layer
    np.save(f'{args.output_dir}/models/history_fit_replica_'+str(i)+'.npy',history) 
    
    final_predictions = model.predict(kin3_norm)
    np.save(f'{args.output_dir}/models/final_predictions_replica_{i}.npy', final_predictions)
    
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

    
    os.makedirs(f'{args.output_dir}/images', exist_ok=True)
    plt.savefig(f'{args.output_dir}/images/image.png')

pseudo = get_data()  

print(pseudo)

kin = np.dstack((pseudo['k'], pseudo['QQ'] , pseudo['xB'], pseudo['t'], pseudo['phi']))
kin = kin.reshape(kin.shape[1:]) # loss inputs

print(kin)

for i in range(0, NUM_OF_REPLICAS):
    start = time.time()
    fit_replica(i, pseudo)
    print("Run Time: ", (time.time() - start)/60, "min")
    
print(sys.path)

# Calculate average MAPE values after all processing
average_mape_reh = np.mean(mape_reh)
average_mape_ree = np.mean(mape_ree)
average_mape_rehtilde = np.mean(mape_rehtilde)
average_mape_dvcs = np.mean(mape_dvcs)
average_mape_total = np.mean(mape_total)

# Load the configurations CSV
config_df = pd.read_csv('configurations.csv')

# Ensure MAPE columns exist
for col in ['mape_reh', 'mape_ree', 'mape_rehtilde', 'mape_dvcs', 'mape_total']:
    if col not in config_df.columns:
        config_df[col] = np.nan

# Convert string representation of list back to list
try:
    nodes_per_layer = ast.literal_eval(args.nodes_per_layer)
except ValueError:
    print("Error: nodes_per_layer argument must be a valid list representation.")
    sys.exit(1)

all_predictions = []

# Load predictions for each replica
for i in range(NUM_OF_REPLICAS):
    predictions = np.load(f'{args.output_dir}/models/final_predictions_replica_{i}.npy')
    all_predictions.append(predictions)
comparison_plots_dir = f'{args.output_dir}/comparison_plots'
os.makedirs(comparison_plots_dir, exist_ok=True)
print(comparison_plots_dir)

all_predictions = np.array(all_predictions)  # Shape should be (NUM_OF_REPLICAS, num_samples, 4)

# Real CFF values (assuming these are available in your dataset)
real_cffs = np.array([pseudo['ReH'], pseudo['ReE'], pseudo['ReHtilde'], pseudo['dvcs']]).transpose()

# Function to plot and save histograms
def plot_and_save_histogram(predictions, real_values, label, title, filename):
    plt.figure()
    plt.hist(predictions, bins=50, alpha=0.5, label=f'Predicted {label}')
    plt.hist(real_values, bins=50, alpha=0.5, label=f'Actual {label}')
    plt.xlabel(f'{label} Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(comparison_plots_dir, filename))
    plt.close()

# Plot and save histograms for each CFF component
plot_and_save_histogram(all_predictions[:, :, 0].flatten(), real_cffs[:, 0], 'ReH', 'Histogram of Predicted vs. Actual ReH', 'ReH_comparison.png')
plot_and_save_histogram(all_predictions[:, :, 1].flatten(), real_cffs[:, 1], 'ReE', 'Histogram of Predicted vs. Actual ReE', 'ReE_comparison.png')
plot_and_save_histogram(all_predictions[:, :, 2].flatten(), real_cffs[:, 2], 'ReHtilde', 'Histogram of Predicted vs. Actual ReHtilde', 'ReHtilde_comparison.png')
plot_and_save_histogram(all_predictions[:, :, 3].flatten(), real_cffs[:, 3], 'DVCS', 'Histogram of Predicted vs. Actual DVCS', 'DVCS_comparison.png')

output_dir = args.output_dir
csv_output_path = f'{output_dir}/model_performance_metrics.csv'
os.makedirs(output_dir, exist_ok=True)

# Function to plot and save scatter plots with vertical error bars for predicted values and scatter points for actual values
def plot_and_save_scatter_with_error_bars(predictions, real_values, label, title, filename):
    plt.figure(figsize=(10, 6))

    # Convert real values to an array to handle multiple real values correctly
    real_values = np.array(real_values)

    # Calculate the mean and standard deviation of predictions
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)

    # Plot predicted values with error bars
    x_values = np.arange(len(mean_predictions))
    plt.errorbar(x_values, mean_predictions, yerr=std_predictions, fmt='o', label=f'Predicted {label} (mean ± std)', color='blue', alpha=0.7)

    # Plot actual values as scatter points
    plt.scatter(x_values, real_values, color='red', label=f'Actual {label}', alpha=0.9, edgecolors='k', zorder=5)

    plt.xlabel('Data Points')
    plt.ylabel(f'{label} Value')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(comparison_plots_dir, filename))
    plt.close()

# Ensure the directory for saving comparison plots exists
comparison_plots_dir = f'{args.output_dir}/comparison_plots'
os.makedirs(comparison_plots_dir, exist_ok=True)

# Assuming all_predictions and real_cffs are already defined
# all_predictions.shape = (NUM_OF_REPLICAS, num_samples, 4)
# real_cffs.shape = (num_samples, 4)

# Flatten the predictions and real values for comparison
all_predictions_flat = all_predictions.reshape(NUM_OF_REPLICAS, -1, 4)
real_cffs_flat = real_cffs.repeat(NUM_OF_REPLICAS, axis=0)

# Plot and save scatter plots with error bars for each CFF component
plot_and_save_scatter_with_error_bars(all_predictions_flat[:, :, 0], real_cffs[:, 0], 'ReH', 'Scatter Plot of Predicted vs. Actual ReH', 'ReH_scatter.png')
plot_and_save_scatter_with_error_bars(all_predictions_flat[:, :, 1], real_cffs[:, 1], 'ReE', 'Scatter Plot of Predicted vs. Actual ReE', 'ReE_scatter.png')
plot_and_save_scatter_with_error_bars(all_predictions_flat[:, :, 2], real_cffs[:, 2], 'ReHtilde', 'Scatter Plot of Predicted vs. Actual ReHtilde', 'ReHtilde_scatter.png')
plot_and_save_scatter_with_error_bars(all_predictions_flat[:, :, 3], real_cffs[:, 3], 'DVCS', 'Scatter Plot of Predicted vs. Actual DVCS', 'DVCS_scatter.png')


# Assuming all_predictions and real_cffs are already defined
# all_predictions.shape = (NUM_OF_REPLICAS, num_samples, 4)
# real_cffs.shape = (num_samples, 4)

# Flatten the predictions and real values for comparison
all_predictions_flat = all_predictions.reshape(-1, 4)
real_cffs_flat = real_cffs.repeat(NUM_OF_REPLICAS, axis=0)

# Function to compute metrics for a single CFF component
def compute_metrics(predictions, actuals):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    std_error = np.std(actuals - predictions)
    corr = np.corrcoef(actuals, predictions)[0, 1]
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r2,
        'MAPE (%)': mape,
        'Std Error': std_error,
        'Correlation': corr
    }

# Compute metrics for each CFF component
metrics = {
    'Metric': ['MAE', 'RMSE', 'R-squared', 'MAPE (%)', 'Std Error', 'Correlation']
}

for i, label in enumerate(['ReH', 'ReE', 'ReHtilde', 'DVCS']):
    component_metrics = compute_metrics(all_predictions_flat[:, i], real_cffs_flat[:, i])
    metrics[label] = list(component_metrics.values())

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)

# Save to CSV
metrics_df.to_csv(csv_output_path, index=False)

print(f"Performance metrics saved to {csv_output_path}")
# Find the row that matches the configuration exactly
'''mask = (config_df['nodes_per_layer'].apply(ast.literal_eval) == nodes_per_layer) & \
       (config_df['activation_function'] == args.activation) & \
       (config_df['learning_rate'] == args.learning_rate) & \
       (config_df['batch_size'] == args.batch_size) & \
       (config_df['epochs'] == args.epochs) & \
       (config_df['jobs'] == args.jobs) & \
       (config_df['optimizer'] == args.optimizer)

# If a matching row is found, update it
if mask.any():
    index = mask.idxmax()  # Get the index of the first True value
    config_df.at[index, 'mape_reh'] = np.mean(mape_reh)
    config_df.at[index, 'mape_ree'] = np.mean(mape_ree)
    config_df.at[index, 'mape_rehtilde'] = np.mean(mape_rehtilde)
    config_df.at[index, 'mape_dvcs'] = np.mean(mape_dvcs)
    config_df.at[index, 'mape_total'] = np.mean(mape_total)
else:
    print("No matching configuration found to update.")

# Save the updated DataFrame back to CSV
config_df.to_csv('configurations.csv', index=False)'''

















