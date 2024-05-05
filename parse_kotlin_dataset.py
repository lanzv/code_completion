import argparse
import logging
import os
import glob
import json
import git
import shutil
import random
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--project_repo_url', type=str, default='https://github.com/JetBrains/kotlin', help='url of the kotlin github repository')
parser.add_argument('--project_directory', type=str, default='./large_kotlin_project', help='temporary directory of the kotlin project for dataset scraping')
parser.add_argument('--dataset_file', type=str, default="kotlin.json", help='name of the file in the directory ./data of the final dataset')
parser.add_argument('--seed', type=int, help='random seed', default=54)




def preprocess_code(code):
    tokenized_code = code.replace("\n", " <EOL> ")
    tokenized_code = tokenized_code.replace(".", " . ")
    tokenized_code = tokenized_code.replace(",", " , ")
    tokenized_code = tokenized_code.replace("{", " { ")
    tokenized_code = tokenized_code.replace("}", " } ")
    tokenized_code = tokenized_code.replace("(", " ( ")
    tokenized_code = tokenized_code.replace(")", " ) ")
    tokenized_code = tokenized_code.replace("\"", " \" ")
    tokenized_code = tokenized_code.replace(":", " : ")
    tokenized_code = tokenized_code.replace("@", " @ ")
    tokenized_code = tokenized_code.replace("?", " ? ")
    tokenized_code = tokenized_code.split()
    return ["<s>"] + tokenized_code + ["</s>"]


def main(args):
    # clone github repo
    git.Repo.clone_from(args.project_repo_url, args.project_directory)
    logging.info("Repository {} cloned successfully".format(args.project_repo_url))

    # collect all kt code
    kotlin_codes = []
    paths = []
    token_count = 0
    for root, dirs, files in os.walk(args.project_directory):
        for file in files:
            if file.endswith(".kt"):
                with open(os.path.join(root, file), 'r') as f:
                    code = f.read()
                paths.append(os.path.join(root, file))
                kotlin_codes.append(preprocess_code(code))
                token_count += len(kotlin_codes[-1])
    logging.info("The code was preprocessed, overall there are {} files, on average of {} tokens".format(len(kotlin_codes), float(token_count)/len(kotlin_codes)))
    
    # process codes and create dataset
    kotlin_dataset = {}
    indices = list(range(len(kotlin_codes)))
    random.shuffle(indices)
    for file_id, i in enumerate(indices):
        kotlin_dataset[file_id] = {"path": paths[i], "code": kotlin_codes[i]}
    with open("./data/{}".format(args.dataset_file), "w") as jsonFile:
        json.dump(kotlin_dataset, jsonFile)
    logging.info("The dataset was created and stored to ./data/{} file".format(args.dataset_file))

    # remove cloned repository
    shutil.rmtree(args.project_directory)
    logging.info("The cloned directory {} was deleted".format(args.project_directory))

if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)