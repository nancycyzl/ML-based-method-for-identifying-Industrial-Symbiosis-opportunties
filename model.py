import torch
import torch.nn as nn
import torch.nn.functional as F


class ISClassifier(nn.Module):
    def __init__(self, company_vector_size, resource_vector_size, hidden_size, last_dim=1, dropout_rate=0):
        super(ISClassifier, self).__init__()

        # MLP for company vectors
        self.company_mlp = nn.Sequential(
            nn.Linear(company_vector_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # MLP for resource vectors
        self.resource_mlp = nn.Sequential(
            nn.Linear(resource_vector_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # MLP for concatenated vectors
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),  # x2 because of concatenation
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, last_dim)  # Output
        )

        # Initialize model parameters
        self.init_weights()

    def forward(self, company_vectors, resource_vectors):
        vector1 = self.company_mlp(company_vectors)
        vector2 = self.resource_mlp(resource_vectors)
        combined_vector = torch.cat((vector1, vector2), dim=1)
        output = self.final_mlp(combined_vector)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

