import logging


def evaluate(gold_data, predictions):
    """
    Part of the official evaluation script (CodeXGLUE)
    gold_data: list of lines, one line ~ string of gold tokens separated by spaces
    predictions: list of lines, one line ~  string of prediced tokens separated by spaces
    """
    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = 0
    correct = 0.0
    for pred, gt in zip(predictions, gold_data):
        pred = pred.split()
        gt = gt.split()
        assert len(pred) == len(gt), f"Sequence length of prediction and answer are not equal, {len(pred)}: {len(gt)}"
        for x, y in zip(pred, gt):
            if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                total += 1
                if x == y:
                    correct += 1
    
    logger.info(f"Total {total} tokens, accuracy: {round(correct/total*100, 2)}")

    return float(correct)/total