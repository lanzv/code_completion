import argparse
import logging
import json
import random
from src.datasets import PythonDataset, KotlinDataset
from src.evaluate import evaluate
from src.llm_models import LLMWrapper
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='python')
parser.add_argument('--model', type=str, default='ClinicalBERT')
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--seed', type=int, help='random seed', default=54)






MODELS = {
    "phi": lambda x: LLMWrapper("../../models/phi-1_5"),
    "stablecode": lambda x: LLMWrapper("../../models/stablecode-completion-alpha-3b"),
    "starcoder2": lambda x: LLMWrapper("../../models/starcoder2-3b")
}

DATASETS = {
    "python": lambda x: PythonDataset(dataset_name="code_x_glue_cc_code_completion_token"),
    "kotlin": lambda x: KotlinDataset(dataset_path="")
}


def main(args):
    if not args.model in MODELS:
        logging.error("The model {} is not supported".format(args.model))
        return
    if not args.dataset in DATASETS:
        logging.error("The dataset {} is not supported".format(args.dataset))
        return
    
    # Load dataset
    dataset = DATASETS[args.dataset]()
    logging.info("{} dataset was loaded successfully".format(args.dataset))

    # Load model and predict
    model = MODELS[args.model]()
    logging.info("{} model was loaded successfully".format(args.model))
    if args.train:
        model.train(dataset.train, dataset.dev)
        logging.info("{} model was trained successfully".format(args.model))
    predictions = model.predict(dataset.test)
    logging.info("predictions were generated")
    
    # Evaluate
    accuracy = evaluate(dataset.test_gold_data, predictions)
    logging.info("Final accuracy on test dataset: {} %".format())
    scores = {
        "{}_{}".format(args.dataset, args.model):
        {
            "accuracy": accuracy
        }
    }
    final_json = json.dumps(scores, indent = 4) 
    print(scores)


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)