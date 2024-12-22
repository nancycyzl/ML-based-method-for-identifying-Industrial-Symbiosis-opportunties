import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class ISDataset(Dataset):
    def __init__(self, filepath, mode="mixed", more_neg=False):
        # 3 labels: 0 - produce, 1 - need, 2 - neither
        self.mode = mode

        self.more_neg_no = 0
        if more_neg:
            if self.mode == "produce":
                self.more_neg_no = 512
            if self.mode == "need":
                self.more_neg_no = 587

        self.full_df = pd.read_csv(filepath)
        self.label_dict = self.set_label_dict()
        self.data = self.process_data(self.full_df, mode=mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        company_description = self.data.iloc[idx]['isic_description']
        resource_description = self.data.iloc[idx]['resource_description']
        label = self.data.iloc[idx]['label']
        return company_description, resource_description, label

    def process_data(self, full_df, mode):
        if mode == "produce":
            # consider bp and sample the same amount from negative samples
            # for EI, add negative sampling (HE produce 512)
            produce_df = full_df[full_df["label"] == "produce"]
            neg_df = full_df[full_df["label"] == "neither"]
            if neg_df.shape[0] > 0:
                neg_sample_df = neg_df.sample(n=produce_df.shape[0]+self.more_neg_no, random_state=42)
                final_df = pd.concat([produce_df, neg_sample_df], ignore_index=True)
            else:
                final_df = produce_df
        elif mode == "need":
            # consider input and sample the same amount from negative samples
            # for EI, add negative sampling (HE need 587)
            input_df = full_df[full_df["label"] == "need"]
            neg_df = full_df[full_df["label"] == "neither"]
            if neg_df.shape[0] > 0:
                neg_sample_df = neg_df.sample(n=input_df.shape[0]+self.more_neg_no, random_state=42)
                final_df = pd.concat([input_df, neg_sample_df], ignore_index=True)
            else:
                final_df = input_df
        elif mode == "mixed":
            # in order to make dataset balanced, all three classes have num(produce)
            produce_df = full_df[full_df["label"] == "produce"]  # smallest amount
            input_df = full_df[full_df["label"] == "need"]
            neg_df = full_df[full_df["label"] == "neither"]
            input_sample_df = input_df.sample(n=produce_df.shape[0])
            neg_sample_df = neg_df.sample(n=produce_df.shape[0])
            final_df = pd.concat([produce_df, input_sample_df, neg_sample_df], ignore_index=True)
        else:
            final_df = full_df

        final_df = self.convert_label(final_df)

        return final_df

    def set_label_dict(self):
        # if produce mode: neither 0, produce 1
        # if need mode: neither 0, need 1,
        # if mixed mode: neither 0, produce 1, need 2
        if self.mode == "mixed":
            label_dict = {"neither": 0, "produce": 1, "need": 2}
        elif self.mode == "produce":
            label_dict = {"neither": 0, "produce": 1}
        else:
            label_dict = {"neither": 0, "need": 1}
        return label_dict

    def convert_label(self, df):
        df.loc[:, 'label'] = df['label'].replace(self.label_dict)
        return df

    def get_details(self):
        return self.label_dict, self.data["label"].value_counts()


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    # for E5
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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


def description_to_vector(descriptions, tokenizer, model, mode="GPT2"):

    if mode == "GPT2":
        # # print("Initializing gpt2 tokenizer and model...")
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # tokenizer.pad_token = tokenizer.eos_token
        # model = GPT2Model.from_pretrained('gpt2')
        # # print("Done!")

        inputs = tokenizer(descriptions, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return embeddings

    elif mode == "E5":
        # # https://huggingface.co/intfloat/e5-base-v2, embedding dimension = 768
        # tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        # model = AutoModel.from_pretrained('intfloat/e5-base-v2')

        # for E5, add "query: " as instructed by the official documents
        descriptions = ["query: " + description for description in descriptions]
        # Tokenize the input texts
        batch_dict = tokenizer(descriptions, max_length=512, padding=True, truncation=True, return_tensors='pt')

        model.eval()
        with torch.no_grad():
            outputs = model(**batch_dict)

        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # [bs, 768]
        return embeddings

    elif mode == "BERT":
        inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Compute embeddings
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return embeddings

    elif mode == "Nomic":
        embeddings = model.encode(descriptions, max_length=512, show_progress_bar=False)
        return torch.tensor(embeddings)

    elif mode == "Jina":
        embeddings = model.encode(descriptions, max_length=512, show_progress_bar=False)
        return torch.tensor(embeddings)


def test_embedding(tokenizer, model, mode):
    print("Test embedding 2 sentences using {}..".format(mode))
    sentences = ["I like apples.", "I like bananas"]
    embeds = description_to_vector(sentences, tokenizer, model, mode=mode)
    print(type(embeds))
    print(embeds.shape)
