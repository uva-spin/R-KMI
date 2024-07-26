import pandas as pd
import subprocess
import os

config_df = pd.read_csv('configurations.csv')

def create_and_submit_job(index, row):
    job_dir = f"jobs/config_{index}"
    os.makedirs(job_dir, exist_ok=True)

    script_content = f"""#!/bin/bash
#SBATCH --job-name=model_train
#SBATCH --output={job_dir}/output.txt
#SBATCH --error={job_dir}/error.txt
#SBATCH --time=05:00:00
#SBATCH --partition=standard
#SBATCH -A spinquest

source activate tf-2.7

python fixed_layers_model_train.py --nodes_per_layer '{row['nodes_per_layer']}' --activation "{row['activation_function']}" --learning_rate {row['learning_rate']} --batch_size {row['batch_size']} --epochs {row['epochs']} --optimizer {row['optimizer']} --output_dir {job_dir}
    """

    script_filename = f"{job_dir}/run_job.sh"
    with open(script_filename, 'w') as file:
        file.write(script_content)
    os.chmod(script_filename, 0o755)
    print(script_content)
    subprocess.run(["sbatch", script_filename], capture_output=True, text=True)

# Using iterrows to iterate over DataFrame rows and include the index
for index, row in config_df.iterrows():
    create_and_submit_job(index, row)
