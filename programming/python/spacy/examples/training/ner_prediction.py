import copy
import os, re
import spacy
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ner_data_generation import ENT_NAMES, DATASET, split_to_train_eval, label_ents

dir_path = os.path.dirname(os.path.realpath(__file__))
nlp = spacy.load(os.path.join(dir_path, 'output/model-best'))


## =========== Evaluation ===========
def prepare_yTruth_yPred(msgs:list[set]):
    yTruth_yPred_pairs_ents:dict[str, list[tuple]] = {}
    for ent_name in ENT_NAMES:
        yTruth_yPred_pairs_ents[ent_name] = []
    idx_print_control = 0
    for text, annot in msgs:
        msg_ent_preds:set[tuple] = set()
        doc_pred = nlp(text)
        for ent_pred in doc_pred.ents:
            msg_ent_preds.add((ent_pred.start_char, ent_pred.end_char, ent_pred.label_))
        for yTruth in annot["entities"]:
            start, end, label = yTruth
            if yTruth in msg_ent_preds:
                yTruth_yPred_pairs_ents[label].append((1,1))
                msg_ent_preds.remove(yTruth)
            else:
                if label in ENT_NAMES:
                    yTruth_yPred_pairs_ents[label].append((0,1))
        # the remainings are false
        for ent_pred in msg_ent_preds:
            start, end, label = ent_pred
            if label in ENT_NAMES:
                yTruth_yPred_pairs_ents[label].append((0,1))
        idx_print_control += 1
        if idx_print_control % 3 == 0:
            print("\nPreds: ")
            for ent_pred in doc_pred.ents:
                print("{}: {}".format( \
                ent_pred.label_, ent_pred.text), end="; ")
            print("\nTruths: ")
            for ent_truth_start, ent_truth_end, ent_truth_label in annot["entities"]:
                print("{}: {}".format( \
                ent_truth_label, text[ent_truth_start:ent_truth_end]), end="; ")

            print("\n")
    return yTruth_yPred_pairs_ents

def compute_eval_metrics(yTruth_yPred_pairs_ents:dict[str, list[tuple]]):
    for label, yTruth_yPred_pairs in yTruth_yPred_pairs_ents.items():
        if len(yTruth_yPred_pairs) == 0:
            print(f"{label}: empty prediction result, skipped")
            continue

        y_truth_labels = [yTruth for yTruth, yPreds in yTruth_yPred_pairs]
        y_pred_labels = [yPreds for yTruth, yPreds in yTruth_yPred_pairs]

        y_truth_pred_same_count = 0
        for yTruth, yPreds in yTruth_yPred_pairs:
            y_truth_pred_same_count += 1 if yTruth == yPreds else 0
        accuracy = float(y_truth_pred_same_count) / len(yTruth_yPred_pairs)
        precision, recall, f1, _ = precision_recall_fscore_support(y_truth_labels, y_pred_labels, average='weighted')
        print(f"{label} Accuracy: {accuracy}")
        print(f"{label} Precision: {precision}")
        print(f"{label} Recall: {recall}")
        print(f"{label} F1 Score: {f1}")

data = label_ents(DATASET)
TRAIN_DATA, EVAL_DATA = split_to_train_eval(data)
yTruth_yPred_pairs_ents = prepare_yTruth_yPred(EVAL_DATA)
compute_eval_metrics(yTruth_yPred_pairs_ents)
