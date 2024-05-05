import logging


def evaluate(gold_data, predictions):
    """
    Evaluate the code completion datataset using the official 
    (or more likely "official") evaluation script "provided" by CodeXGLUE
    (not commented, not structured, but part of the official dataset code ... )

    Adapted to our project goal and project structure, but evaluated the same way

    Parameters
    ----------
    gold_data : list of lists of tokens
        list of gold tokenized codes
    predictions : list of lists of tokens
        list of predicted tokenized codes
    Returns
    -------
    accuracy : float
        accuracy of correct tokens (ignoring <s>, </s>, <EOL>, <pad>)
    """
    assert len(predictions) == len(gold_data), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = 0
    correct = 0.0
    for pred, gt in zip(predictions, gold_data):
        assert len(pred) == len(gt), f"Sequence length of prediction and answer are not equal, {len(pred)}: {len(gt)}"
        for x, y in zip(pred, gt):
            if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                total += 1
                if x == y:
                    correct += 1
    
    logging.info(f"Total {total} tokens, accuracy: {round(correct/total*100, 2)}")

    return float(correct)/total