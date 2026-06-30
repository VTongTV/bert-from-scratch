def entity_f1(preds, labels, label_map=None):
    pred_entities = set()
    label_entities = set()
    i = 0
    while i < len(preds):
        if preds[i] != 0:
            start = i
            tag = preds[i]
            while i < len(preds) and preds[i] == tag:
                i += 1
            pred_entities.add((start, i, tag))
        else:
            i += 1
    i = 0
    while i < len(labels):
        if labels[i] != 0:
            start = i
            tag = labels[i]
            while i < len(labels) and labels[i] == tag:
                i += 1
            label_entities.add((start, i, tag))
        else:
            i += 1
    tp = len(pred_entities & label_entities)
    if len(pred_entities) == 0 or len(label_entities) == 0:
        return 0.0
    precision = tp / len(pred_entities)
    recall = tp / len(label_entities)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
