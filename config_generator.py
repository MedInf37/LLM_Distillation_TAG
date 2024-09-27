import os

def create_folder_and_write_files(folder_name, file_name, file_content):
    """
    Create a folder and write text files with prewritten text.

    Args:
    folder_name (str): The name of the folder to create.
    file_prefix (str): The prefix for the filenames.
    text_content (str): The content to write in each file.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file_path = os.path.join(folder_name, f"{file_name}")
    with open(file_path, 'w') as file:
        file.write(file_content)

def write_files(custom_name):
    ########################################################################################################################
    for_server = True # TODO change if config etc is for server
    if for_server:
        absolute_path = f"/home/yourfirstname.yourlastname/experiment/{custom_name}/"
    else:
        absolute_path = f"./{custom_name}/"

    lm_name = "distilbert-base-uncased" # TODO change if you want other model
    max_layer = 5 if lm_name == "distilbert-base-uncased" else 9
    dataset_name = "ogbn-products" # TODO change if you want other dataset
    only_title = False # TODO change if you do not want full text
    noisy = True # TODO change if you want noisy text
    train_robust_model = True if noisy else False # TODO change first boolean if you want robust lm

    # write config files
    config_folder_name = f"config/{custom_name}_configs"

    if train_robust_model: # write config file 0
        config_file0_name = f"train_robust_lm_{custom_name}_config.yaml"
        absolute_path_config0 = absolute_path + f"output_dir_train_robust_lm_{custom_name}"
        path_to_pretrained_classifier = "/home/yourfirstname.yourlastname/experiment/classifier_with_standard_bert_products/model_checkpoints/classifier_with_standard_bert_products.pth" # TODO change if you want other pretrained classifier
        dropout_rate = 0.1 # TODO change if you want
        learning_rate = 1.3476469778946654e-05 # TODO change if you want
        max_epoch = 2 # TODO change if you want
        custom_name_robust_model = f"robust_{lm_name}_for_noisy_{dataset_name}_max_epoch_{max_epoch}"
        config_file0_content = f"""absolute_path: {absolute_path_config0}
model_name: {lm_name}
dataset_name: {dataset_name}
path_to_pretrained_classifier: {path_to_pretrained_classifier}
dropout_rate: {dropout_rate}
learning_rate: {learning_rate}
noisy: {noisy}
save_model_small: true
max_epoch: {max_epoch}
custom_name: {custom_name_robust_model}"""
        create_folder_and_write_files(config_folder_name, config_file0_name, config_file0_content)

    # write config file 1
    config_file1_name = f"save_embedding_{custom_name}_config.yaml"
    absolute_path_config1 = absolute_path + f"output_dir_save_{custom_name}_embedding"
    path_to_save_file_config1 = f"{custom_name}_embedding.npy"
    if train_robust_model:
        path_to_model = f"{absolute_path_config0}/model_checkpoints/{custom_name_robust_model}"
    else:
        path_to_model = "/home/yourfirstname.yourlastname/experiment/gpt2_distillation_products/tune_hyperparameters_gpt2_products/model_checkpoints/lr_0.0002187023895664096" # TODO change if you want other model
    config_file1_content = f"""absolute_path: {absolute_path_config1}
use_lightning_checkpoint: false
lm_name: {lm_name}
dataset_name: {dataset_name}
only_title: {only_title}
noisy: {noisy}
device: cuda
path_to_model: {path_to_model}
path_to_save_file: {path_to_save_file_config1}"""
    create_folder_and_write_files(config_folder_name, config_file1_name, config_file1_content)

    # write config file 2
    config_file2_name = f"distill_experiment_for_{custom_name}_config.yaml"
    absolute_path_config2 = absolute_path + f"output_dir_{custom_name}_experiment"
    path_to_teacher_embedding = f"{absolute_path_config1}/{path_to_save_file_config1}"
    config_file2_content = f"""absolute_path: {absolute_path_config2}
model_name: {lm_name}
dataset_name: {dataset_name}
only_title: {only_title}
noisy: {noisy}
path_to_teacher_embedding: {path_to_teacher_embedding}
min_layers: 1
max_layers: {max_layer}"""
    create_folder_and_write_files(config_folder_name, config_file2_name, config_file2_content)

    # write config file 3
    config_file3_name = f"distill_embedding_{custom_name}_config.yaml"
    absolute_path_config3 = absolute_path + f"output_dir_distill_embedding_{custom_name}"
    config_file3_content = f"""absolute_path: {absolute_path_config3}
file_path: {absolute_path_config2}/model_checkpoints_test
model_name: {lm_name}
dataset_name: {dataset_name}
only_title: {only_title}
noisy: {noisy}"""
    create_folder_and_write_files(config_folder_name, config_file3_name, config_file3_content)

    # write config file 4
    config_file4_name = f"distill_baseline_{custom_name}_config.yaml"
    absolute_path_config4 = absolute_path + f"output_dir_baseline_{custom_name}"
    config_file4_content = f"""absolute_path: {absolute_path_config4}
path_to_original_embedding: {path_to_teacher_embedding}
path_to_embeddings: {absolute_path_config3}

dataset_name: {dataset_name}

graph_model:
    hidden_channels: 256
    num_layers: 1
    num_heads: 8
    dropout: 0.33
    edge_dropout: 0.48
    input_dropout: 0.35
    input_norm: False
    different_embedding: None
    learning_rate: 0.00152
    weight_decay: 0.00055
    trial_number: 9999999
    batch_size: 256
    test: true
    label_smoothing: 0.01
    number_of_sampled_neighbors: 50"""
    create_folder_and_write_files(config_folder_name, config_file4_name, config_file4_content)

    train_on_clean_too = True # TODO change if you want
    if noisy and train_on_clean_too:
        clean_config_folder_name = config_folder_name + "/on_clean_text_configs"
        clean_absolute_path = absolute_path + "performance_on_clean_text/"

        # write clean config file 1
        clean_config_file1_name = f"save_embedding_{custom_name}_clean_config.yaml"
        clean_absolute_path_config1 = clean_absolute_path + f"output_dir_save_{custom_name}_clean_embedding"
        clean_path_to_save_file_config1 = f"{custom_name}_clean_embedding.npy"
        config_file1_content = f"""absolute_path: {clean_absolute_path_config1}
use_lightning_checkpoint: false
lm_name: {lm_name}
dataset_name: {dataset_name}
only_title: {only_title}
noisy: {False}
device: cuda
path_to_model: {path_to_model}
path_to_save_file: {clean_path_to_save_file_config1}"""
        create_folder_and_write_files(clean_config_folder_name, clean_config_file1_name, config_file1_content)

        # clean config file 2 is not necessary since distillation already happened

        # write clean config file 3
        clean_config_file3_name = f"distill_embedding_{custom_name}_clean_config.yaml"
        clean_absolute_path_config3 = clean_absolute_path + f"output_dir_distill_embedding_{custom_name}_clean"
        clean_config_file3_content = f"""absolute_path: {clean_absolute_path_config3}
file_path: {absolute_path_config2}/model_checkpoints_test
model_name: {lm_name}
dataset_name: {dataset_name}
only_title: {only_title}
noisy: {False}"""
        create_folder_and_write_files(clean_config_folder_name, clean_config_file3_name, clean_config_file3_content)

        # write config file 4
        clean_config_file4_name = f"distill_baseline_{custom_name}_config.yaml"
        clean_absolute_path_config4 = clean_absolute_path + f"output_dir_baseline_{custom_name}_clean"
        clean_config_file4_content = f"""absolute_path: {clean_absolute_path_config4}
path_to_original_embedding: {os.path.join(clean_absolute_path_config1, clean_path_to_save_file_config1)}
path_to_embeddings: {clean_absolute_path_config3}

dataset_name: {dataset_name}

graph_model:
    hidden_channels: 256
    num_layers: 1
    num_heads: 8
    dropout: 0.33
    edge_dropout: 0.48
    input_dropout: 0.35
    input_norm: False
    different_embedding: None
    learning_rate: 0.00152
    weight_decay: 0.00055
    trial_number: 9999999
    batch_size: 256
    test: true
    label_smoothing: 0.01
    number_of_sampled_neighbors: 50"""
        create_folder_and_write_files(clean_config_folder_name, clean_config_file4_name, clean_config_file4_content)

    ########################################################################################################################
    # write .sub files
    sub_files_folder_name = f"multiple_{custom_name}_jobs"

    if train_robust_model:
        # write job0.sub
        job0_file_name = "job0.sub"
        job0_file_content = f"""executable = execute.sh
arguments = --method train_lm --config {config_folder_name}/{config_file0_name}
transfer_input_files = transfolder/
log = execute_train_{custom_name_robust_model}.log
output = stdout_train_{custom_name_robust_model}.txt
error = stderr_train_{custom_name_robust_model}.txt
request_gpus = 1
request_memory = 40 GB
+MaxRuntime = 345600
queue"""
        create_folder_and_write_files(sub_files_folder_name, job0_file_name, job0_file_content)

    # write job1.sub
    job1_file_name = "job1.sub"
    job1_file_content = f"""executable = execute.sh
arguments = --method save_embedding --config {config_folder_name}/{config_file1_name}
transfer_input_files = transfolder/
log = execute_save_{custom_name}_embedding.log
output = stdout_save_{custom_name}_embedding.txt
error = stderr_save_{custom_name}_embedding.txt
request_gpus = 1
request_memory = 40 GB
+MaxRuntime = 345600
queue"""
    create_folder_and_write_files(sub_files_folder_name, job1_file_name, job1_file_content)

    # write job2.sub
    job2_file_name = "job2.sub"
    job2_file_content = f"""executable = execute.sh
arguments = --method distill_experiment --config {config_folder_name}/{config_file2_name}
transfer_input_files = transfolder/
log = execute_{custom_name}_experiment.log
output = stdout_{custom_name}_experiment.txt
error = stderr_{custom_name}_experiment.txt
request_gpus = 1
request_memory = 40 GB
+MaxRuntime = 345600
queue"""
    create_folder_and_write_files(sub_files_folder_name, job2_file_name, job2_file_content)

    # write job3.sub
    job3_file_name = "job3.sub"
    job3_file_content = f"""executable = execute.sh
arguments = --method distill_embeddings --config {config_folder_name}/{config_file3_name}
transfer_input_files = transfolder/
log = execute_distill_embedding_{custom_name}.log
output = stdout_distill_embedding_{custom_name}.txt
error = stderr_distill_embedding_{custom_name}.txt
request_gpus = 1
request_memory = 40 GB
+MaxRuntime = 345600
queue"""
    create_folder_and_write_files(sub_files_folder_name, job3_file_name, job3_file_content)

    # write job4.sub
    job4_file_name = "job4.sub"
    job4_file_content = f"""executable = execute.sh
arguments = --method distill_baseline --config {config_folder_name}/{config_file4_name}
transfer_input_files = transfolder/
log = execute_{custom_name}_baseline.log
output = stdout_{custom_name}_baseline.txt
error = stderr_{custom_name}_baseline.txt
request_gpus = 1
request_memory = 40 GB
+MaxRuntime = 345600
queue"""
    create_folder_and_write_files(sub_files_folder_name, job4_file_name, job4_file_content)

    if noisy and train_on_clean_too:
        clean_sub_files_folder_name = sub_files_folder_name + "/on_clean_text_jobs"

        # write job1.sub
        clean_job1_file_name = "job1.sub"
        clean_job1_file_content = f"""executable = execute.sh
arguments = --method save_embedding --config {clean_config_folder_name}/{clean_config_file1_name}
transfer_input_files = transfolder/
log = execute_save_{custom_name}_clean_embedding.log
output = stdout_save_{custom_name}_clean_embedding.txt
error = stderr_save_{custom_name}_clean_embedding.txt
request_gpus = 1
request_memory = 40 GB
+MaxRuntime = 345600
queue"""
        create_folder_and_write_files(clean_sub_files_folder_name, clean_job1_file_name, clean_job1_file_content)

        # write job3.sub
        clean_job3_file_name = "job3.sub"
        clean_job3_file_content = f"""executable = execute.sh
arguments = --method distill_embeddings --config {clean_config_folder_name}/{clean_config_file3_name}
transfer_input_files = transfolder/
log = execute_distill_clean_embedding_{custom_name}.log
output = stdout_distill_clean_embedding_{custom_name}.txt
error = stderr_distill_clean_embedding_{custom_name}.txt
request_gpus = 1
request_memory = 40 GB
+MaxRuntime = 345600
queue"""
        create_folder_and_write_files(clean_sub_files_folder_name, clean_job3_file_name, clean_job3_file_content)

        # write job4.sub
        clean_job4_file_name = "job4.sub"
        clean_job4_file_content = f"""executable = execute.sh
arguments = --method distill_baseline --config {clean_config_folder_name}/{clean_config_file4_name}
transfer_input_files = transfolder/
log = execute_{custom_name}_clean_baseline.log
output = stdout_{custom_name}_clean_baseline.txt
error = stderr_{custom_name}_clean_baseline.txt
request_gpus = 1
request_memory = 40 GB
+MaxRuntime = 345600
queue"""
        create_folder_and_write_files(clean_sub_files_folder_name, clean_job4_file_name, clean_job4_file_content)

    ########################################################################################################################
    # write jobs.dag file
    dag_folder_name = "."
    dag_file_name = f"{custom_name}_jobs.dag"
    if train_robust_model and not train_on_clean_too:
        dag_file_content = f"""# Job definitions
JOB JOB0 {sub_files_folder_name}/job0.sub
JOB JOB1 {sub_files_folder_name}/job1.sub
JOB JOB2 {sub_files_folder_name}/job2.sub
JOB JOB3 {sub_files_folder_name}/job3.sub
JOB JOB4 {sub_files_folder_name}/job4.sub

# Job dependencies
PARENT JOB0 CHILD JOB1
PARENT JOB1 CHILD JOB2
PARENT JOB2 CHILD JOB3
PARENT JOB3 CHILD JOB4"""
    elif train_robust_model and train_on_clean_too:
        dag_file_content = f"""# Job definitions
JOB JOB0 {sub_files_folder_name}/job0.sub
JOB JOB1 {sub_files_folder_name}/job1.sub
JOB JOB2 {sub_files_folder_name}/job2.sub
JOB JOB3 {sub_files_folder_name}/job3.sub
JOB JOB4 {sub_files_folder_name}/job4.sub
JOB JOB5 {clean_sub_files_folder_name}/job1.sub
JOB JOB6 {clean_sub_files_folder_name}/job3.sub
JOB JOB7 {clean_sub_files_folder_name}/job4.sub

# Job dependencies
PARENT JOB0 CHILD JOB1
PARENT JOB1 CHILD JOB2
PARENT JOB2 CHILD JOB3
PARENT JOB3 CHILD JOB4
PARENT JOB4 CHILD JOB5
PARENT JOB5 CHILD JOB6
PARENT JOB6 CHILD JOB7"""
    else:
        dag_file_content = f"""# Job definitions
JOB JOB1 {sub_files_folder_name}/job1.sub
JOB JOB2 {sub_files_folder_name}/job2.sub
JOB JOB3 {sub_files_folder_name}/job3.sub
JOB JOB4 {sub_files_folder_name}/job4.sub

# Job dependencies
PARENT JOB1 CHILD JOB2
PARENT JOB2 CHILD JOB3
PARENT JOB3 CHILD JOB4"""
    create_folder_and_write_files(dag_folder_name, dag_file_name, dag_file_content)
    ########################################################################################################################

def generate_job_file(num_jobs, sub_files_folder_name="topic", file_name="job.dag"):
    with open(file_name, "w") as file:
        # Write job definitions
        file.write("# Job definitions\n")
        for i in range(1, num_jobs + 1):
            file.write(f"JOB JOB{i} {sub_files_folder_name}/job{i}.sub\n")

        # Write job dependencies
        file.write("\n# Job dependencies\n")
        for i in range(1, num_jobs):
            file.write(f"PARENT JOB{i} CHILD JOB{i + 1}\n")

def write_in_bulk(custom_name, names: list):
    absolute_path = f"/home/yourfirstname.yourlastname/experiment/{custom_name}/" # TODO change if other
    
    config_folder_name = f"config/{custom_name}_configs"
    sub_files_folder_name = f"multiple_{custom_name}_jobs"

    for i, name in enumerate(names):
        lm_name = "bert-base-uncased"
        lm_name = "distilbert-base-uncased" if "distilbert" in name else lm_name
        lm_name = "gpt2" if "gpt2" in name else lm_name
        if "products" in name:
            dataset_name = "ogbn-products"
        else:
            dataset_name = "ogbn-arxiv"
        # write config file
        config_file_name = f"save_{name}_noisy_standard_embedding_config.yaml"
        absolute_path = f"/home/yourfirstname.yourlastname/experiment/{custom_name}/"
        config_file1_content = f"""absolute_path: {absolute_path}
lm_name: {lm_name}
use_standard: true
dataset_name: {dataset_name}
noisy: true
path_to_save_file: {name}_noisy_standard_embedding.npy"""
        create_folder_and_write_files(config_folder_name, config_file_name, config_file1_content)

        # write job.sub
        job_file_name = f"job{i+1}.sub"
        memory = 20 if dataset_name == "ogbn-arxiv" else 40
        job_file_content = f"""executable = execute.sh
arguments = --method save_embedding --config {config_folder_name}/{config_file_name}
transfer_input_files = transfolder/
log = execute_save_{name}_noisy_standard_embedding.log
output = stdout_save_{name}_noisy_standard_embedding.txt
error = stderr_save_{name}_noisy_standard_embedding.txt
request_gpus = 1
request_memory = {memory} GB
+MaxRuntime = 345600
queue"""
        create_folder_and_write_files(sub_files_folder_name, job_file_name, job_file_content)
    
    generate_job_file(num_jobs=len(names), sub_files_folder_name=sub_files_folder_name, file_name=f"{custom_name}_jobs.dag")

if __name__ == "__main__":
    input("Did you remember to change everything you want?")
    write_files(custom_name="distilbert_noisy_products") # TODO change file name



    # names = ["bert_arxiv",
    #         #  "bert_only_title",
    #          "distilbert_arxiv",
    #         #  "distilbert_only_title",
    #          "gpt2_arxiv",
    #         #  "gpt2_only_title",
    #          "bert_products",
    #          "distilbert_products",
    #          "gpt2_products"]
    # write_in_bulk(custom_name="save_noisy_standard_embeddings", names=names)