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
parser.add_argument('--dataset', type=str, default='python', help='specify code of dataset.. [python, kotlin]')
parser.add_argument('--model', type=str, default='phi', help='specify code of model .. [phi] (eventually others..)')
parser.add_argument('--train', type=bool, default=False, help='True for finetuning model, otherwise False in case only pretrained model is used')
parser.add_argument('--disable_tqdm', type=bool, default=False, help='True for muting the tqdm loading bar')
parser.add_argument('--evaluate_on_python_data', type=bool, default=False, help='not only the args.dataset is measured on the trained model, but also the python dataset is measured')
parser.add_argument('--seed', type=int, help='random seed', default=54)






MODELS = {
    "phi": lambda x: LLMWrapper("../../models/phi-1_5"),
    "stablecode": lambda x: LLMWrapper("../../models/stablecode-completion-alpha-3b"),
    "starcoder2": lambda x: LLMWrapper("../../models/starcoder2-3b")
}

DATASETS = {
    "python": lambda x: PythonDataset(dataset_name="code_x_glue_cc_code_completion_token"),
    "kotlin": lambda x: KotlinDataset(dataset_path="./data/kotlin.json")
}


def main(args):
    if not args.model in MODELS:
        logging.error("The model {} is not supported".format(args.model))
        return
    if not args.dataset in DATASETS:
        logging.error("The dataset {} is not supported".format(args.dataset))
        return
    
    # Load dataset
    dataset = DATASETS[args.dataset](1) # ToDo Remove lambda x and the random parameter 1
    logging.info("{} dataset was loaded successfully".format(args.dataset))

    # Load model and predict
    model = MODELS[args.model](1) # ToDo Remove lambda x and the random parameter 1
    logging.info("{} model was loaded successfully".format(args.model))
    if args.train:
        model.train(dataset.train, dataset.dev, disable_tqdm=args.disable_tqdm)
        logging.info("{} model was trained successfully".format(args.model))
    gold_data, predictions = model.predict(dataset.test, disable_tqdm=args.disable_tqdm)
    logging.info("predictions were generated")

    # Evaluate
    accuracy = evaluate(gold_data, predictions)
    logging.info("Final accuracy on the test dataset: {} %".format(accuracy*100))
    scores = {
        "{}_{}".format(args.dataset, args.model):
        {
            "accuracy": accuracy
        }
    }
    final_json = json.dumps(scores, indent = 4)
    print(final_json)

    
    if args.evaluate_on_python_data: # kind of messy but the task description asks for this
        dataset = DATASETS["python"](1) # ToDo Remove lambda x and the random parameter 1
        logging.info("{} dataset was loaded successfully".format("python"))
        gold_data, predictions = model.predict(dataset.test, disable_tqdm=args.disable_tqdm)
        logging.info("predictions were generated")
        accuracy = evaluate(gold_data, predictions)
        logging.info("Final accuracy on the test dataset: {} %".format(accuracy*100))
        scores = {
            "{}_{}".format("python", args.model):
            {
                "accuracy": accuracy
            }
        }
        final_json = json.dumps(scores, indent = 4)
        print(final_json)


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)