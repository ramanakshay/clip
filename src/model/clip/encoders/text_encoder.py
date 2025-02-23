from torch import nn
from transformers import DistilBertModel, DistilBertConfig


class TextEncoder(nn.Module):
    def __init__(self, from_pretrained, proj_dim):
        super(TextEncoder, self).__init__()
        if from_pretrained:
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else:
            config = DistilBertConfig()
            self.model = DistilBertModel(config)
        output_dim = self.model.config.dim
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(output_dim, proj_dim, bias=False))

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state.mean(dim=1)
        projection = self.projection(last_hidden_state)
        return projection