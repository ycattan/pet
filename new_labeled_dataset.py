import os
import random
import json
import shutil

"""
This script is used to create other sets of 32 labeled datapoints.
To put it differently, it's like recreating other versions of FewGLUE.
The idea is to study the impact it can have on the performance of the final classifier.

"""

def extract_datapoints(input_path, output_path, N) :

    L = []
    f = open(input_path)
    n_lines = sum(1 for _ in f)

    # Draw randomly N datapoints.
    extractions = random.sample(range(1, n_lines), N)
    extractions = sorted(extractions)

    # Extract datapoints from input_path file
    f = open(input_path)
    for k, sentence in enumerate(f) :
        if k in extractions :
            L.append(json.loads(sentence))

    # Write the extracted data into the output file.
    with open(f"{output_path}", "w") as output_file :
        for item in L :
            output_file.write(json.dumps(item) + "\n")

    return


def main(n_set=3, N=32) :
    """
    For each task :

    1. Create n_set subfolders of task dataset named "{task}_{k}" 
    for k in [1, n_set] 

    2. Copy/Paste to this new folder the same unlabeled, validation
    and test set as in the orignial task folder.

    3. Create the new train dataset using N randomly chosen 
    datapoints.

    """

    superglue_path = './data/superglue'
    fewglue_path = './data/fewglue'

    
    for file in os.listdir(superglue_path) :
        d = os.path.join(superglue_path, file)
        if os.path.isdir(d) :

            dataset_name = str(d).split('\\')[-1]

            # We discard AX-b and AX-g datasets
            if dataset_name[0] != 'A' :
                
                for i in range(n_set) :
                    output_folder = f"{fewglue_path}/{dataset_name}/{dataset_name}_{i+1}/"
                    
                    # Create new folder if does not exist yet
                    if not os.path.isdir(output_folder) :
                        os.mkdir(output_folder)

                    # Copy/paste unlabeled, validation and test set
                    # i.e. everything but the train.jsonl file
                    for file in os.listdir(f"{fewglue_path}/{dataset_name}") :
                        if not os.path.isdir(os.path.join(f"{fewglue_path}/{dataset_name}", file)) :
                            if 'train' not in file :
                                source = os.path.join(f"{fewglue_path}/{dataset_name}", file)
                                dest = f"{fewglue_path}/{dataset_name}/{dataset_name}_{i+1}/{file}"
                                shutil.copy(source, dest)

                    # Create train.jsonl by extracting N random examples from initial
                    # train.jsonl file in original superGLUE dataset
                    original_training_set = f"{superglue_path}/{dataset_name}/train.jsonl"
                    new_training_set = f"{fewglue_path}/{dataset_name}/{dataset_name}_{i+1}/train.jsonl"
                    extract_datapoints(input_path=original_training_set,
                                       output_path=new_training_set,
                                       N=N)                        

    return 


#########################
# Execute main() function
#########################
main()



                





