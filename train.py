'''
This script is to train a classifier to predict company-resource relationships.
2 labels: 0 - neither, 1 - produce / need

Input: EI_data1_valid_need.csv (generated from data_preprocess.py and data_preprocess_filter.py)
Note: dataset_mode can be "produce" (binary classification) / "need" (binary) / "mixed" (3 labels)
'''

import torch
from data_utils import ISDataset, description_to_vector, test_embedding
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset, random_split
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from model import ISClassifier
import torch.nn as nn
import tqdm
import argparse
import os
from utils import *
import logging
from evaluation import evaluate_model
import time


def make_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--EI_file', type=str, default="data/EI_data1_valid_need.csv", help="Path to the EI input csv file.")
    parser.add_argument('--output_dir', type=str, default="output_new", help="output directory for saving trained models")
    parser.add_argument('--epochs', type=int, default=20, help="how many epochs to train")
    parser.add_argument('--dataset_mode', type=str, default="produce", help="mixed / produce / need")
    parser.add_argument('--embedding_model', type=str, default="GPT2", choices=["GPT2", "E5", "Nomic", "Jina"])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--l2_lambda', type=float, default=0.0, help='coefficient for the L2 regularization term')

    parser.add_argument('--train_set', type=str, default="EI", help="EI / HE / EIHE")
    parser.add_argument('--train_slice', type=float, default=1, help="Percentage of training data to test influence of data amount")
    parser.add_argument('--train_slice_seed', type=float, default=20, help="random set to sample training data")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--embedding_dim', type=int, default=768, help="embedding dimension for company/waste descriptions")
    parser.add_argument('--hidden_dim', type=int, default=128, help="hidden dimension for company and resource")
    parser.add_argument('--threshold', type=float, default=0.5, help="threshold to determine positive and negative")
    parser.add_argument('--f1_modified', action="store_true", default=False, help="whether use modified F1 score")
    parser.add_argument('--dropout', type=float, default=0, help="dropout factor in classifier")
    parser.add_argument("--scheduler_stepsize", type=int, default=100, help="learning rate scheduler step size")
    parser.add_argument('--dry_run', action="store_true", default=False, help='whether to test whole process')
    parser.add_argument("--load_pretrained", type=str, default="", help="path of trained .pth model if want to resume or do eval only")
    parser.add_argument("--eval_only", action='store_true', default=False, help="random seed for initialization")
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


def train(model, embedding_tokenizer, embedding_model, train_loader, val_loader, args):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=0.5)

    model.to(args.device)
    model.train()  # Set model to training mode
    step_loss_list = []
    epoch_loss_list_train = []
    epoch_loss_list_val = []
    train_accuracy_list = []
    val_accuracy_list = []
    for epoch in range(args.epochs):
        epoch_str = "Training epoch: {}, learning rate: {}".format(epoch, optimizer.param_groups[0]['lr'])
        print(epoch_str)
        logging.info(epoch_str)
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_logits = []
        for company_descriptions, resource_descriptions, labels in tqdm.tqdm(train_loader):

            # company_vectors = description_to_vector(company_descriptions, mode=args.embedding_model).to(args.device)
            # resource_vectors = description_to_vector(resource_descriptions, mode=args.embedding_model).to(args.device)

            company_vectors = description_to_vector(company_descriptions, embedding_tokenizer, embedding_model, mode=args.embedding_model).to(args.device)
            resource_vectors = description_to_vector(resource_descriptions, embedding_tokenizer, embedding_model, mode=args.embedding_model).to(args.device)

            labels = labels.to(args.device).float()

            optimizer.zero_grad()

            # Forward pass
            outputs = model(company_vectors, resource_vectors).squeeze()
            loss = criterion(outputs, labels)

            # Add L2 regularization term
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
            loss = loss + args.l2_lambda * l2_norm

            # Backward and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            # metrics
            total_loss += loss.item()
            step_loss_list.append(loss.item())

            class_pred = logits_to_labels(outputs, args.threshold).cpu().numpy()
            all_predictions.extend(class_pred)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.detach().cpu().numpy())


        # Calculate metrics
        cm = confusion_matrix(all_labels, all_predictions)
        fpr, tpr, _ = roc_curve(all_labels, all_logits)
        roc_auc = auc(fpr, tpr)
        accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_predictions)

        train_accuracy_list.append(accuracy)

        epoch_metrics = "Loss: {:4f}, Accuracy: {:4f}, Precision: {:4f}, Recall: {:4f}, F1: {:4f}, ROC: {:4f}".format(
            total_loss / len(train_loader), accuracy, precision, recall, f1, roc_auc)
        print(epoch_metrics)
        logging.info(epoch_metrics)

        # add val set metrics
        val_acc, _, _, _, _, val_loss = evaluate_model(model, embedding_tokenizer, embedding_model, val_loader, args)
        val_accuracy_list.append(val_acc)
        epoch_loss_list_val.append(val_loss)

        if args.dry_run:
            break

        # save model dict every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(args.output_dir, 'model_state_dict_epoch{}.pth'.format(epoch))
            torch.save(model.state_dict(), save_path)

    save_training_statistics(args.output_dir, step_loss_list, epoch_loss_list_train, epoch_loss_list_val,
                             train_accuracy_list, val_accuracy_list)
    save_arguments(model, args, filename="arguments.txt")


def logits_to_labels(logits, threshold):
    probs = torch.sigmoid(logits)
    labels = (probs >= threshold).int()
    return labels


def split_train_val_test(data, val_size, test_size):
    # train & val+test
    train, data2 = train_test_split(data, test_size=val_size+test_size, random_state=20)
    # split val and test
    val, test = train_test_split(data2, test_size=test_size/(val_size+test_size), random_state=20)

    return train, val, test



def prepare_dataloaders(args):
    # dataset: EI
    # train val test: 70%, 15%, 15%
    print("Preparing datasets...")
    dataset_EI = ISDataset(filepath=args.EI_file, mode=args.dataset_mode, more_neg=False)
    print("EcoInvent data details:", "\n", dataset_EI.get_details())
    # dataset_EI_train, dataset_EI_test = train_test_split(dataset_EI, test_size=0.2, random_state=20)
    dataset_EI_train, dataset_EI_val, dataset_EI_test = split_train_val_test(dataset_EI, val_size=0.15, test_size=0.15)
    print("After splitting train, val, test sizes: ", len(dataset_EI_train), len(dataset_EI_val), len(dataset_EI_test))

    # dataset_HE = ISDataset(filepath=args.HE_file, mode=args.dataset_mode)
    # print("Historical data details:", "\n", dataset_HE.get_details())
    # dataset_HE_train, dataset_HE_test = train_test_split(dataset_HE, test_size=0.2, random_state=42)

    # slice part of training set, for analysis of data amount influence
    if args.train_slice < 1:
        torch.manual_seed(args.train_slice_seed)  # for reproducibility / run multiple times (seed=20, 42, 88)
        total_size = len(dataset_EI_train)
        split_size = int(total_size * args.train_slice)
        dataset_EI_train, _ = random_split(dataset_EI_train, [split_size, total_size - split_size])
        print("Change training size: {}%, num of samples: {}".format(args.train_slice * 100, len(dataset_EI_train)))

    # create dataloader
    # if args.train_set == "EI":
    train_loader = DataLoader(dataset_EI_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_EI_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_EI_test, batch_size=args.batch_size, shuffle=False)
    print("Done!")

    # elif args.train_set == "HE":
    #     dataset_HE_train, dataset_HE_val = train_test_split(dataset_HE_train, test_size=0.2, random_state=42)
    #     train_loader = DataLoader(dataset_HE_train, batch_size=args.batch_size, shuffle=True)
    #     val_loader = DataLoader(dataset_HE_val, batch_size=args.batch_size)
    #     test_loader = DataLoader(dataset_HE_test, batch_size=args.batch_size)
    #
    # else:  # train_set = "EI_HE"
    #     dataset_train = ConcatDataset([dataset_EI_train, dataset_HE_train])
    #     train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    #     val_loader = DataLoader(dataset_EI_test, batch_size=args.batch_size, shuffle=False)
    #     test_loader = DataLoader(dataset_HE_test, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    args = make_argument()
    test_argument(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=os.path.join(args.output_dir, 'logging.log'),  # Logs will be saved to this file
                        filemode='a')  # Use 'a' to append to the file instead of 'w' which overwrites the file

    train_loader, val_loader, test_loader = prepare_dataloaders(args)

    # prepare embedding model
    # load embedding tokenizer and model
    embedding_tokenizer, embedding_model = load_embedding_model(args.embedding_model)
    test_embedding(embedding_tokenizer, embedding_model, args.embedding_model)

    # training
    print("Training model on ", args.device)
    if args.dataset_mode == "mixed":
        model_last_dim = 3
    else:
        model_last_dim = 1  # for produce / need
    model = ISClassifier(company_vector_size=768, resource_vector_size=768, hidden_size=args.hidden_dim, last_dim=model_last_dim)
    if args.load_pretrained:
        model_dict = torch.load(args.load_pretrained)
        model.load_state_dict(model_dict)
        print("Load trained model from {}".format(args.load_pretrained))

    # start training
    if not args.eval_only:
        train_start_time = time.time()
        train(model, embedding_tokenizer, embedding_model, train_loader, val_loader, args)
        logging.info("---" * 20)
        logging.info("Training hours: {}".format((time.time() - train_start_time)/3600))
        print("Training hours: ", (time.time() - train_start_time)/3600)

        # save trained model
        model_save_path = os.path.join(args.output_dir, 'model_state_dict.pth')
        torch.save(model.state_dict(), model_save_path)
        save_result = "All results are saved at: {}".format(args.output_dir)
        print(save_result)
        logging.info(save_result)

    # evaluation - train set
    accuracy, precision, recall, f1, roc_auc, _ = evaluate_model(model, embedding_tokenizer, embedding_model,
                                                                 train_loader, args, note="train set", save_plot=True)
    train_result = f'Train Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC: {roc_auc:.4f}'
    print(train_result)
    logging.info(train_result)

    # evaluation - val set
    accuracy, precision, recall, f1, roc_auc, _ = evaluate_model(model,embedding_tokenizer, embedding_model,
                                                                 val_loader, args, note="val set", save_plot=True)
    val_result = f'Val Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC: {roc_auc:.4f}'
    print(val_result)
    logging.info(val_result)

    # evaluation - test set
    accuracy, precision, recall, f1, roc_auc, _ = evaluate_model(model, embedding_tokenizer, embedding_model,
                                                                 test_loader, args, note="test set", save_plot=True)
    test_result = f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC: {roc_auc:.4f}'
    print(test_result)
    logging.info(test_result)
