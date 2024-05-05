from datasets import load_dataset, Dataset
import json
import logging

class PythonDataset:
    """
    Python code completion dataset
    Load json file of Kotlin code, the final dataset contains following rows
        'id': unique id number
        'path': path of the original file from given kotlin project
        'code': list of code tokens

    public parameters
        self.train
        self.dev
        self.test
    """
    def __init__(self, dataset_name):
        dataset = load_dataset(dataset_name, "python")
        splited_train = dataset["train"].train_test_split(test_size=0.05, shuffle = False)
        self.train = splited_train["train"]
        self.dev = splited_train["test"]
        self.test = dataset["test"]



class KotlinDataset:
    """
    Kotlin code completion dataset
    Load json file of Kotlin code, the final dataset contains following rows
        'id': unique id number
        'path': path of the original file from given kotlin project
        'code': list of code tokens

    public parameters
        self.train
        self.dev
        self.test
    """
    def __init__(self, dataset_path):
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        data = {
            "id": [],
            "path": [],
            "code": []
        }
        for file_id in dataset:
            data["id"].append(file_id)
            data["path"].append(dataset[file_id]["path"])
            data["code"].append(dataset[file_id]["code"])
        dataset = Dataset.from_dict(data)
        dataset = dataset.train_test_split(test_size=0.33, shuffle = False) #it's already shuffled
        splited_train = dataset["train"].train_test_split(test_size=0.05, shuffle = False) # it's already shuffled
        self.train = splited_train["train"]
        self.dev = splited_train["test"]
        self.test = dataset["test"]
        logging.info("Kotlin dataset is prepared, train: {}, dev: {}, test: {}".format(len(self.train), len(self.dev), len(self.test)))