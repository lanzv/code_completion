from datasets import load_dataset

class PythonDataset:
    def __init__(self, dataset_name):
        dataset = load_dataset(dataset_name, "python")
        self.train = dataset["train"].train_test_split(test_size=0.05)["train"]
        self.dev = dataset["train"].train_test_split(test_size=0.05)["test"]
        self.test = dataset["test"]



class KotlinDataset:
    def __init__(self, dataset_path):
        self.train = None
        self.test = None
        self.dev = None
        self.test_gold_data = None