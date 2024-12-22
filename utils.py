import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def load_txt_file_line(filename):
    with open(filename, 'r') as f:
        lines = [line.strip()for line in f.readlines()]   # still str yet
    return lines


def load_json_file(filename):
    with open(filename, 'r') as f:
        content = json.load(f)
    return content


def plot_step_loss(loss_list, filepath):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, linestyle='-', color='b')
    plt.title('Loss per step')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filepath)


def plot_step_loss_smooth(loss_list, filepath, window_size=5):
        weights = np.ones(window_size) / window_size
        smooth_loss = np.convolve(loss_list, weights, mode='valid')
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(window_size-1, len(loss_list)), smooth_loss, linestyle='-', color='r')
        plt.title('Smooth loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(filepath)


def plot_epoch_loss(loss_list_train, loss_list_val, filepath):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list_train, linestyle='-', color='b', label='train')
    plt.plot(loss_list_val, linestyle='-', color='r', label='val')
    plt.title('Train and val loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)


def plot_epoch_accuracy(train_accuracy_list, val_accuracy_list, filepath):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracy_list, linestyle='-', color='b', label='train')
    plt.plot(val_accuracy_list, linestyle='-', color='r', label='val')
    plt.title('Train and val accuracy per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)


def plot_confusion_matrix(cm, filename):
    # Use seaborn to create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename)


def plot_metrics(accuracies, precisions, recalls, f1s, aucs, thresholds, filename, note=""):
    plt.figure(figsize=(10, 6))

    # Create a mask to exclude negative values
    def filter_negative(x, y):
        return [(tx, ty) for tx, ty in zip(x, y) if ty >= 0]

    # Filter data for each metric
    acc_points = filter_negative(thresholds, accuracies)
    prec_points = filter_negative(thresholds, precisions)
    rec_points = filter_negative(thresholds, recalls)
    f1_points = filter_negative(thresholds, f1s)
    auc_points = filter_negative(thresholds, aucs)

    # Unzip points back into separate x and y lists
    if acc_points:
        acc_x, acc_y = zip(*acc_points)
        plt.plot(acc_x, acc_y, linestyle='-', color='b', label='accuracy')
    if prec_points:
        prec_x, prec_y = zip(*prec_points)
        plt.plot(prec_x, prec_y, linestyle='-', color='r', label='precision')
    if rec_points:
        rec_x, rec_y = zip(*rec_points)
        plt.plot(rec_x, rec_y, linestyle='-', color='y', label='recall')
    if f1_points:
        f1_x, f1_y = zip(*f1_points)
        plt.plot(f1_x, f1_y, linestyle='-', color='g', label='f1')
    if auc_points:
        auc_x, auc_y = zip(*auc_points)
        plt.plot(auc_x, auc_y, linestyle='-', color='orange', label='auc')

    plt.title('Metrics ' + note)
    plt.xlabel('threshold')
    plt.ylabel('value')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


def plot_roc_curve(fpr, tpr, roc_auc, filename):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)


def calculate_metrics(labels, predictions, f1_modified=False):
    # if len(cm) == 2:
    #     TN = cm[0][0]
    #     FN = cm[1][0]
    #     FP = cm[0][1]
    #     TP = cm[1][1]
    #     accuracy = (TN + TP) / (TN + FN + FP + TP)
    #     precision = TP / (TP + FP)
    #     recall = TP / (TP + FN)
    #     f1 = 2 * precision * recall / (precision + recall)
    #     return accuracy, precision, recall, f1
    # else:
    #     # to do, for each class
    #     return 0, 0, 0, 0
    TN = FN = FP = TP = 0
    if len(set(labels)) <= 2:
        # binary classification
        for l, p in zip(labels, predictions):
            if l == 0 and p == 0:
                TN += 1
            elif l == 0 and p == 1:
                FP += 1
            elif l == 1 and p == 1:
                TP += 1
            else:
                FN += 1
        accuracy = safe_divide(TN + TP, TN + FN + FP + TP)
        precision = safe_divide(TP, TP + FP)
        recall = safe_divide(TP, TP + FN)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        if f1_modified:
            predict_positive_rate = safe_divide(TP + FP, TN + FN + FP + TP)
            f1 = safe_divide(recall ** 2, predict_positive_rate)

        return accuracy, precision, recall, f1
    else:
        return 0, 0, 0, 0


def safe_divide(nominator, denominator):
    if denominator > 0:
        return nominator / denominator
    else:
        return -1


def test_argument(args):
    # for dataset mode
    valid_mode = ['mixed', 'produce', 'need']
    if args.dataset_mode not in valid_mode:
        raise Exception("Dataset mode is not valid, it should be mixed/produce/need.")
    # output_dir exists or not
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def save_arguments(model, args, filename="arguments.txt"):
    with open(os.path.join(args.output_dir, filename), 'w') as file:
        for arg in vars(args):
            file.write(f'{arg}: {getattr(args, arg)}\n')
        file.write('\n' + '-' * 40 + '\n\n')

        for layer in model.children():
            print(layer, file=file)


def save_training_statistics(save_folder, loss_list, epoch_loss_list_train, epoch_loss_list_val,
                             train_accuracy_list, val_accuracy_list):
    plot_step_loss(loss_list, filepath=os.path.join(save_folder, "training step loss.png"))
    plot_step_loss_smooth(loss_list, filepath=os.path.join(save_folder, "training step loss smooth.png"))

    if len(epoch_loss_list_train) > 0:
        plot_epoch_loss(epoch_loss_list_train, epoch_loss_list_val, filepath=os.path.join(save_folder, "train and val epoch loss.png"))

    plot_epoch_accuracy(train_accuracy_list, val_accuracy_list, filepath=os.path.join(save_folder, "train and val accuracy.png"))


def remove_2dlist_duplicates(list2d):
    return list(map(list, set(map(tuple, list2d))))


def check_existing_file(filepath):
    if os.path.exists(filepath):
        # Split the filepath into folder, base, and extension
        folder, filename = os.path.split(filepath)
        base, extension = os.path.splitext(filename)

        # Initialize a counter for the suffix
        counter = 1
        # Generate a new filename with a counter suffix until an unused name is found
        new_filename = os.path.join(folder, f"{base}{counter}{extension}")
        while os.path.exists(new_filename):
            counter += 1
            new_filename = os.path.join(folder, f"{base}{counter}{extension}")

        return new_filename
    else:
        # If the file doesn't exist, return the original filepath
        return filepath


def load_embedding_model(embedding_model):
    if embedding_model == "GPT2":
        # dim = 768
        from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
        tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
        model_gpt2 = GPT2Model.from_pretrained('gpt2')

        return tokenizer_gpt2, model_gpt2

    elif embedding_model == "E5":
        # https://huggingface.co/intfloat/e5-base-v2
        # dim = 768
        from transformers import AutoTokenizer, AutoModel
        tokenizer_e5 = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        model_e5 = AutoModel.from_pretrained('intfloat/e5-base-v2')

        return tokenizer_e5, model_e5

    elif embedding_model == "BERT":
        from transformers import BertTokenizer, BertModel
        tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
        model_bert = BertModel.from_pretrained('bert-base-uncased')

        return tokenizer_bert, model_bert

    elif embedding_model == "Nomic":
        # https://huggingface.co/nomic-ai/nomic-embed-text-v1?utm_source=chatgpt.com
        # dim = 768
        from sentence_transformers import SentenceTransformer

        model_nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

        return None, model_nomic

    elif embedding_model == "Jina":
        # https://huggingface.co/jinaai/jina-embeddings-v2-base-en?utm_source=chatgpt.com
        # dim = 768
        from transformers import AutoModel

        model_jina = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

        return None, model_jina

