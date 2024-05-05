import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluate import evaluate
import logging
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

class LLMWrapper:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, sep_token="<EOL>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    def train(self, train, dev, epochs=3, batch_size=2, learning_rate=5e-4, disable_tqdm=False):
        """
        Fine-tune the give model

        Parameters
        ----------
        train : Dataset
            training dataset where each sample contains rows ["id", "path", "code"]
        dev : Dataset
            dev dataset where each sample contains rows ["id", "path", "code"]
        epochs : int
            number of training epochs
        batch_size : int
            batch size of the training (be careful due to GB RAM quota)
        learning_rate : float
            learning rate of the trainng
        disable_tqdm : boolean
            True for showing tqdm loading bar, False in order to mute it
        """
        train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=self.__collate_tokenize)

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 10, num_training_steps = epochs * (len(train_dataloader) // batch_size))

        for i, epoch in enumerate(range(epochs)):
            for _, batch in enumerate(tqdm(train_dataloader, disable=disable_tqdm)):
                input_ids = batch.to(next(self.model.parameters()).device)
                attention_mask = (batch != self.tokenizer.pad_token_id).bool()
                labels = batch.to(next(self.model.parameters()).device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                self.model.zero_grad()
            # Evaluate dev after each epoch
            gd, prds = self.predict(dev, disable_tqdm)
            logging.info("Epoch: {} \t Dev accuracy: {}".format(i, evaluate(gd, prds)))
            self.model.train()
        
    
    def predict(self, test, batch_size=8, disable_tqdm=False):
        """
        Complete code in real time, and tokenize gold data as well as those predicted ones

        Parameters
        ----------
        test : Dataset
            test dataset where each sample contains rows ["id", "path", "code"]
        batch_size : int
            batch size of the training (be careful due to GB RAM quota)
        disable_tqdm : boolean
            True for showing tqdm loading bar, False in order to mute it
        Returns
        -------
        gold_data : list of lists of tokens
            ground truth data, list of all tokenized code samples 
        final_predictions: 
            model predictions, list of all tokenized code samples 
        """
        self.model.eval()
        eval_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, collate_fn=self.__collate_tokenize)
        final_predictions = []
        gold_data = []
        for step, batch in enumerate(tqdm(eval_dataloader, disable=disable_tqdm)):
            attention_mask = (batch != self.tokenizer.pad_token_id).bool()
            input_ids = batch.to(next(self.model.parameters()).device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred_scores = outputs[0]
                predicted_ids = pred_scores.argmax(-1)
            for pred, gold in zip(predicted_ids, batch):
                pred = pred.cpu().tolist()[:-1]
                gold = gold.cpu().tolist()[1:]
                final_predictions.append(self.tokenizer.convert_ids_to_tokens(pred))
                gold_data.append(self.tokenizer.convert_ids_to_tokens(gold))
        return gold_data, final_predictions

    def __collate_tokenize(self, batch):
        code_batch = [' '.join(sample["code"]) for sample in batch]
        return self.tokenizer(code_batch, padding='longest', truncation=True, return_tensors='pt').input_ids.to(next(self.model.parameters()).device)
