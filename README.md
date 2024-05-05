# Code Completion Task

This is the JetBrains Code Completion Interview Task. Our goal is to get the interview. To be more precise, we want to measure performance of the LLM Phi-1.5 model on the Python and Kotlin Code Completion datasets. But first we need to create the Kotlin dataset. This is a Readme that will briefly introduce the problem, datasets and experiments. The Research on this task itself is at the beggining a should be researched and be debugged in more detail in the future.


Definition of the code completion task:
- input: 1...(N-1)th context tokens
- output: Nth token
- iterativally till the there is no gold data code token

Structure of the Readme

- [Usage](#usage)
    - How to parse dataset and how to run the experiment pipeline
    - Eventually how to extend the generalized experiment pipeline for other models/datasets
- [Related Works](#related-works)
    - brief introduction to datasets and models we are going to use in this work
    - we believe there are many of other works (for instance for the kotlin) that should also be mentioned there
- [Datasets](#datasets)
    - How the Kotlin dataset is created + some basic stats of both python/kotlin datasets
- [Methods](#methods)
    - what methods we used
- [Experiments and Results](#experiments-and-results)
    - results of experiments
- [Conclusion](#conclusion)


## Usage

### Prepare kotlin dataset
The final dataset will be stored at ```./data directory``` with the name of ```--dataset_file```

In the following example, the JetBrains/kotlin project is scraped, the ./large_kotlin_project temporary directory is used for that, and the final kotlin dataset is placed in the ```./data/kotlin.json``` file

```sh
python parse_kotlin_dataset.py --project_repo_url 'https://github.com/JetBrains/kotlin' --project_directory './large_kotlin_project' --dataset_file 'kotlin.json'
```

### Run inference/fine-tuning
- Choose the pretrained model (right now we support [```phi```, ```stablecode```, ```starcode2```], theirs checkpoints/pretrained models had to be stored in paths [```../../models/phi-1_5```, ```../../models/stablecode-completion-alpha-3b```, ```../../models/starcoder2-3b```] respectivaly. Feel free to go to the ```run_experiment.py``` and add a new model to ```MODELS``` dictionary, eventually modify paths to them (you can also use the huggingface model code)).
- Choose the dataset code, either ```kotlin``` or ```python```. 
- If you want to mute all loading bars, use ```--disable_tqdm True```
- If you want to do only inference, evaluate on the dataset without finetuning, remove the ```--train True``` tag from the command.
- If you want to also evaluate python data (no matter what ```--dataset``` was specified) on pretrained model, use the ```--evaluate_on_python_data``` tag.

```sh
python run_experiment.py --model phi --dataset kotlin --disable_tqdm True --train True --evaluate_on_python_data True
```

#### Slurm
In order to use slurm, just check the ```./scripts``` directory
```sh
sbatch ./scripts/kpthi_train_python.sh
```

## Related Works
The code completion task can bring us better user experience for not only programmers and IDE users. Several works have been done in order to move forward in this domain. Very important part of the code completion is the [CodeXGLUE](https://microsoft.github.io/CodeXGLUE/) dataset consisting of Java and Kotlin parts. They provide the token-level code completion dataset as well as the line-level code completion dataset. Their current official leader board on the python token-level datset looks as follows:

| Model           | Accuracy |
|-----------------|----------|
| PyCoder         | 76.93    |
| CodeGPT-adapted | 76.60    |
| CodeGPT         | 76.58    |
| GPT-2(12L)      | 75.90    |

The PyCoder could be then used as something like baseline for us. On the other hand there is no known upperbound (for instance human-based) for us at this moment which should be also developed to see the potential of the research.

There are also other models trained to complete code - such as Phi-1.5 (1.3B), Phi-2 (2.7B), StarCode2 (3B, 7B, 15B), or StableCode Completion Alpha 3B. 

## Datasets
In this work we use two datasets - one for Python, one for Kotlin.
Since there are not many of low-resource language datasets, we have to have prepared scraping pipeline to collect and parse own dataset. This is what we are going to do for kotlin.

### Python Dataset
In this work, we are going to use the token-level code completion [CodeXGLUE](https://microsoft.github.io/CodeXGLUE/) dataset.
|                       | Train  | Dev  | Test  |
|-----------------------|--------|------|-------|
| # code samples        | 95 000 | 5 000| 50 000|

### Kotlin Dataset
The https://github.com/JetBrains/kotlin project is scraped and parsed to create the Kotlin dataset. First we found all files with the ```.kt``` extension. We replaced all new lines by ```<EOL>```. Then we splited the kotlin code into list of tokens. We considered all these characters as separated tokens: ```[.,{}()\:@?]``` and we added extra ```<s>``` at the beggining and ```</s>``` at the end.

Then we include the `id` and `path` in the dataset as well. The dataset contains following rows:
 - `id` : unique kotlin file id
 - `path` : path to the original kotlin file
 - `code` : tokenized kotlin code

We provide basic statistics of our Kotlin dataset
 - Size of the dataset
    
    |                       | Train  | Dev  | Test  |
    |-----------------------|--------|------|-------|
    | # code samples        | 34 798 | 1 832| 18 043|
 - The average number of tokens in one code sampe: ***432.04***
 - Number of unique tokens: ***341 884***
 - Average token length (character based): ***4.32***
 - Number of `<EOL>` tokens: ***14 520 705***, which is ***265.59*** of `<EOL>` tokens per code sample

## Methods

In this work we considered pretrained model [Phi-1.5](https://huggingface.co/microsoft/phi-1_5), the LLM specialized for Python coding. Then, we fine-tuned the model for the Kotlin coding.

For the training we found out that the model is overfitted after the first epoch. So we used one epoch, with learning rate equal to 5e-4. 

We refer to the Phi-1.5 finetuned on the Kotlin dataset as **Phi-1.5 KFT**. The Phi-1.5 finetuned on the Python dataset is called **Phi-1.5 PFT**. The basic pretrained **Phi-1.5** is refered as **Phi-1.5**.


## Experiments and Results

| Model       | Python Acc (%) | Kotlin Acc(%) |
|-------------|----------------|---------------|
| PyCoder     | 76.93          | -             |
| Phi-1.5     | 71.48          | 62.49         |
| Phi-1.5 KFT | 0.0022         | 0.78          |
| Phi-1.5 PFT | 0.0022         | -             |

For experiments we used the state-of-the-art **PyCoder** of the official CodeXGLUE leaderboard as basline that we were not able to outperform. On the other hand, the pretrained model Phi-1.5 itself is very close to that. The interesting part is that even though the model is mainly pretrained to Python task, it performs very well also on the Kotlin dataset. 

As we can see, the Finetuning process will totally destroy the pretrained model. Right now it is hard to say why. One possible explanation is that the Kotlin/Python datasets are that small that the model will overfit itself into the training dataset, especially when Phi-1.5 was trained on the general Python code, not the tokenized one as in the way the CodeXGLUE (or our Kotlin dataset) are. Of course in our evaluation pipeline could be mistake or we have to find better hyperparameters for training - some debugging and testing and additional research have to be done first - let's leave it for the future work right now.

Note that the Python results measured using the Phi-1.5 KFT and Phi-1.5 PFT models are even worse than the KFT model on Kotlin data. Either the Kotlin dataset is more consistant either the model is more confused then when it tries to relearn the same language on a more specific task.


## Conclusion
In this work, we created Kotlin Code Completion dataset and provide its brief anlyses. We measured the Phi-1.5 preatrained LLM on the Python and Kotlin Code Completion datasets and found out, that the pretrained LLM itself has very good peformance on the Python dataset. Furthermore, it has very good results for the Kotlin dataset as well even though the model is not focused on the Kotlin language. The finetuning doesn't work well for now - to be able to say why we need to do better Python/Kotlin datasets analyzes, double-check the training process and debug it properly on dev data, eventually to get larger kotlin datasets.