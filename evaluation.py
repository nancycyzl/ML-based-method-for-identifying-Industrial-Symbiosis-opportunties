import torch
from data_utils import ISDataset, description_to_vector
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from model import ISClassifier
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import numpy as np
import tqdm
import argparse
import os
from utils import *
import logging



def make_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default="output/train_EI_data1_produce_epoch50/model_state_dict.pth", help="Path to the trained model.")
    parser.add_argument('--dataset_mode', type=str, default="produce", help="produce / need")
    parser.add_argument('--embedding_model', type=str, default="GPT2", choices=["GPT2", "E5", "Nomic", "Jina"])
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--hidden_dim', type=int, default=128, help="hidden dimension for company and resource")
    parser.add_argument('--output_dir', type=str, default="", help="output directory for saving trained models")
    parser.add_argument('--threshold', type=float, default=0.5, help="threshold to determine positive and negative")
    parser.add_argument('--f1_modified', action="store_true", default=False, help="whether use modified F1 score")
    args = parser.parse_args()
    return args


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


def evaluate_model(model, embedding_tokenizer, embedding_model, data_loader, args, note="", save_plot=False):

    model.to(args.device)
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_labels = []
    all_logits = []
    total_loss = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for company_descriptions, resource_descriptions, labels in tqdm.tqdm(data_loader):

            # company_vectors = description_to_vector(company_descriptions, mode=args.embedding_model).to(args.device)
            # resource_vectors = description_to_vector(resource_descriptions, mode=args.embedding_model).to(args.device)

            company_vectors = description_to_vector(company_descriptions, embedding_tokenizer, embedding_model,
                                                    mode=args.embedding_model).to(args.device)
            resource_vectors = description_to_vector(resource_descriptions, embedding_tokenizer, embedding_model,
                                                     mode=args.embedding_model).to(args.device)

            labels = labels.to(args.device)

            outputs = model(company_vectors, resource_vectors).squeeze()
            if outputs.dim() == 0:  # Check if it's a scalar
                outputs = outputs.unsqueeze(0)  # Convert to 1D

            class_pred = logits_to_labels(outputs, args.threshold).cpu().numpy()

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            all_predictions.extend(class_pred)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_logits)
    roc_auc = auc(fpr, tpr)
    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_predictions, f1_modified=args.f1_modified)

    if save_plot:
        plot_confusion_matrix(cm, filename=os.path.join(args.output_dir, note+"_confusion_matrix.png"))
        plot_roc_curve(fpr, tpr, roc_auc, filename=os.path.join(args.output_dir, note + "_ROC.png"))

    return accuracy, precision, recall, f1, roc_auc, total_loss/len(data_loader)


def prepare_dataset(mode):
    print("Preparing dataloader...")

    dataset_HE_noneg = ISDataset(filepath="data/HE_data_noneg.csv", mode=mode)

    dataset_HE_data1_noneg = ISDataset(filepath="data/HE_data1_noneg.csv", mode=mode)
    dataset_HE_data1_noneg_valid = ISDataset(filepath="data/HE_data1_noneg_valid.csv", mode=mode)
    dataset_HE_data1_noneg_invalid = ISDataset(filepath="data/HE_data1_noneg_invalid.csv", mode=mode)

    dataset_HE_data1_cpc_noneg= ISDataset(filepath="data/HE_data1_cpc_noneg.csv", mode=mode)
    dataset_HE_data1_cpc_noneg_valid = ISDataset(filepath="data/HE_data1_cpc_noneg_valid.csv", mode=mode)
    dataset_HE_data1_cpc_noneg_invalid = ISDataset(filepath="data/HE_data1_cpc_noneg_invalid.csv", mode=mode)

    # dataset_HE_lvl2_noneg = ISDataset(filepath="data/HE_data1_lvl2_noneg.csv", mode="produce")
    # dataset_HE_lvl23_noneg = ISDataset(filepath="data/HE_data1_lvl23_noneg.csv", mode="produce")
    #
    # dataset_EI_train, dataset_EI_test = train_test_split(dataset_EI, test_size=0.2, random_state=42)
    # dataset_HE_train, dataset_HE_test = train_test_split(dataset_HE, test_size=0.2, random_state=42)
    #
    # dataset_train = ConcatDataset([dataset_EI_train, dataset_HE_train])

    datasets = [dataset_HE_noneg, dataset_HE_data1_noneg, dataset_HE_data1_noneg_valid, dataset_HE_data1_noneg_invalid,
                dataset_HE_data1_cpc_noneg, dataset_HE_data1_cpc_noneg_valid, dataset_HE_data1_cpc_noneg_invalid]
    names = ["dataset_HE_noneg", "dataset_HE_data1_noneg", "dataset_HE_data1_noneg_valid", "dataset_HE_data1_noneg_invalid",
             "dataset_HE_data1_cpc_noneg", "dataset_HE_data1_cpc_noneg_valid", "dataset_HE_data1_cpc_noneg_invalid"]

    assert len(datasets) == len(names), "dataset list and name list do not match"

    return datasets, names


def logits_to_labels(logits, threshold):
    probs = torch.sigmoid(logits)
    labels = (probs >= threshold).int()
    return labels


def prepare_trained_models():
    model_path_list = ["output/train_EIHE_produce_epoch50/model_state_dict_epoch0.pth",
                       "output/train_EIHE_produce_epoch50/model_state_dict_epoch10.pth",
                       "output/train_EIHE_produce_epoch50/model_state_dict_epoch20.pth",
                       "output/train_EIHE_produce_epoch50/model_state_dict_epoch30.pth",
                       "output/train_EIHE_produce_epoch50/model_state_dict_epoch40.pth",
                       "output/train_EIHE_produce_epoch50/model_state_dict.pth"]

    model_name_list = ["produce_epoch0", "produce_epoch10", "produce_epoch20",
                       "produce_epoch30", "produce_epoch40", "produce_epoch50"]

    assert len(model_path_list) == len(model_name_list), "model path list and name list do not match"

    return model_path_list, model_name_list


def evaluate_multi_dataset(args, dataset_list, name_list):
    # measure performance on multiple dataset with one model

    model_str = "Loading model dict from {}".format(args.model_path)
    logging.info(model_str + '\n')

    model = ISClassifier(company_vector_size=768, resource_vector_size=768, hidden_size=args.hidden_dim, last_dim=1)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)

    # dataset_list, name_list = prepare_dataset(mode=args.dataset_mode)

    print("Make predictions and evaluation for", name_list, end="\n\n")
    for dataset, name in zip(dataset_list, name_list):
        print("Evaluating dataset:", name)
        logging.info("Evaluating dataset: {}".format(name))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        accuracy, precision, recall, modified_f1, roc_auc, _ = evaluate_model(
            model, dataloader, args, note="{}_thre{}".format(name, args.threshold), save_plot=True)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        result_str = "{} (total {}): Accuracy {:4f}, Precision {:4f}, Recall {:4f}, F1 {:4f}, modified_F1 {:4f}, AUC {:.4f}".format(
            name, len(dataset), accuracy, precision, recall, f1, modified_f1, roc_auc)
        print(result_str)
        logging.info(result_str + "\n")

    print("Result saved at: ", args.output_dir)


def evaluate_multi_model(args, model_paths, model_names):
    # measure performance on one dataset with a series of trained models

    # test dataset
    name = "HE_data1_cpc_test"
    logging.info("Evaluating dataset: {}".format(name))
    dataset = ISDataset(filepath="data/HE_data1_cpc.csv", mode=args.dataset_mode)
    dataset_HE_train, dataset_HE_test = train_test_split(dataset, test_size=0.2, random_state=42)
    dataloader = DataLoader(dataset_HE_test, batch_size=args.batch_size, shuffle=False)

    # model_paths, model_names = prepare_trained_models()
    for model_path, model_name in zip(model_paths, model_names):
        model_str = "Loading model {} from {}".format(model_name, model_path)
        print(model_str)
        logging.info(model_str + '\n')

        model = ISClassifier(company_vector_size=768, resource_vector_size=768, hidden_size=args.hidden_dim, last_dim=1)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        # evaluate
        accuracy, precision, recall, modified_f1, roc_auc, _ = evaluate_model(
            model, dataloader, args, note="{}_thre{}".format(model_name, args.threshold), save_plot=True)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        result_str = "{} (total {}): Accuracy {:4f}, Precision {:4f}, Recall {:4f}, F1 {:4f}, modified_F1 {:4f}, AUC {:.4f}".format(
            name, len(dataset), accuracy, precision, recall, f1, modified_f1, roc_auc)
        print(result_str)
        logging.info(result_str + "\n")

    print("Result saved at: ", args.output_dir)


def evaluate_multi_threshold(args):
    # loading model
    model_str = "Loading model dict from {}".format(args.model_path)
    logging.info(model_str + '\n')

    model = ISClassifier(company_vector_size=768, resource_vector_size=768, hidden_size=args.hidden_dim, last_dim=1)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)

    # prepare test dataset
    name = "HE_data"
    logging.info("Evaluating dataset: {}".format(name))
    dataset = ISDataset(filepath="data/HE_data.csv", mode=args.dataset_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # define threshold
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # evaluate
    for threshold in thresholds:
        thre_str = "Evaluating when threshold is {}".format(threshold)
        logging.info(thre_str + "\n")
        print(thre_str)
        args.threshold = threshold
        accuracy, precision, recall, modified_f1, roc_auc, _ = evaluate_model(
            model, dataloader, args, note="thre{}".format(threshold), save_plot=True)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        result_str = "{} (total {}): Accuracy {:4f}, Precision {:4f}, Recall {:4f}, F1 {:4f}, modified_F1 {:4f}, AUC {:.4f}".format(
            name, len(dataset), accuracy, precision, recall, f1, modified_f1, roc_auc)
        print(result_str)
        logging.info(result_str + "\n")


if __name__ == '__main__':

    # args preparation
    args = make_argument()
    if args.output_dir == "":
        args.output_dir = os.path.join(os.path.dirname(args.model_path), "eval_thre{}".format(args.threshold))
    test_argument(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # making logging file
    logging_file_path = check_existing_file(os.path.join(args.output_dir, "eval_logging.log"))
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=logging_file_path,
                        filemode='w')

    # evaluate: 2 modes
    # evaluate_multi_dataset(args, *prepare_dataset(mode=args.dataset_mode))
    # evaluate_multi_model(args, *prepare_trained_models())
    # evaluate_multi_threshold(args)
