import pandas as pd
import subprocess
import os

# Load configurations
config_df = pd.read_csv('configurations.csv')
print("Loaded configurations CSV")

def create_and_submit_job(row):
    # Extract parameters
    nodes = row['nodes']
    layers = row['layers']
    activation_function = row['activation_function']
    learning_rate = row['learning_rate']
    batch_size = row['batch_size']
    epochs = row['epochs']
    jobs = row['jobs']
    optimizer = row['optimizer']
    
    # Directory for saving job scripts and output
    job_dir = f"jobs/nodes_{nodes}_layers_{layers}_activation_{activation_function}"
    os.makedirs(job_dir, exist_ok=True)
    
    # Generate batch script content
    script_content = f"""#!/bin/sh
#SBATCH --job-name=model_train_{nodes}_{layers}
#SBATCH --time=05:00:00
#SBATCH --output={job_dir}/output_%j.txt
#SBATCH --error={job_dir}/error_%j.txt
#SBATCH --partition=standard
#SBATCH -A spinquest_standard

source activate tf-2.7

python model_train.py --nodes {nodes} --layers {layers} --activation {activation_function} \\
                      --learning_rate {learning_rate} --batch_size {batch_size} --epochs {epochs} \\
                      --optimizer {optimizer} --output_dir {job_dir}
    """
    
    # Write script to a batch file
    script_filename = f"{job_dir}/run_job.sh"
    with open(script_filename, 'w') as file:
        file.write(script_content)
        
    # Set the script to be executable
    os.chmod(script_filename, 0o755)
    
    # Submit job using sbatch and do not wait for it to finish
    process = subprocess.Popen(["sbatch", script_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    print("STDOUT:", stdout)
    print("STDERR:", stderr)

# Apply the function to each row in the DataFrame
config_df.apply(create_and_submit_job, axis=1)
