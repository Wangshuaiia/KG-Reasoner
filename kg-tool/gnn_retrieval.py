import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

class GraphSAGE(nn.Module):
    """
    A simplified GraphSAGE implementation:
    - Single-layer aggregation
    - Mean aggregator
    """
    def __init__(self, in_dim, out_dim):
        super(GraphSAGE, self).__init__()
        # We'll concatenate [node vector + mean of neighbor vectors],
        # then project it into a new dimension.
        self.fc = nn.Linear(in_dim * 2, out_dim)

    def forward(self, node_embeds, neighbors_embeds_list):
        """
        node_embeds: [num_nodes, in_dim]
            The embedding of each node (encoded by BERT).
        neighbors_embeds_list: a Python list of length 'num_nodes'.
            Each element i is a Tensor of shape [num_neighbors_i, in_dim],
            containing the embeddings of all neighbors for node i.

        Return: [num_nodes, out_dim]
            The updated node embedding after neighbor aggregation.
        """
        out = []
        for i in range(len(node_embeds)):
            if neighbors_embeds_list[i].shape[0] > 0:
                neighbor_mean = neighbors_embeds_list[i].mean(dim=0)  # [in_dim]
            else:
                neighbor_mean = torch.zeros_like(node_embeds[i])

            # Concatenate current node vector and neighbor mean
            concat_vec = torch.cat([node_embeds[i], neighbor_mean], dim=-1)  # [2*in_dim]
            out_i = self.fc(concat_vec)  # [out_dim]
            out.append(out_i)

        # Stack them back into a [num_nodes, out_dim] tensor
        out = torch.stack(out, dim=0)
        return out


class MyModel(nn.Module):
    """
    Overall model:
    1) BERT text encoding
    2) GraphSAGE neighbor aggregation
    3) MLP for classification (binary in this example)
    """
    def __init__(self,
                 bert_name='bert-base-uncased',  # English BERT model
                 bert_hidden_dim=768,           # dimension for BERT-base
                 gnn_hidden_dim=128,            # dimension after GNN
                 num_classes=2):                # binary classification
        super(MyModel, self).__init__()

        # 1) BERT encoder
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.bert = BertModel.from_pretrained(bert_name)

        # 2) GraphSAGE
        self.gnn = GraphSAGE(bert_hidden_dim, gnn_hidden_dim)

        # 3) Classifier (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def encode_texts(self, texts, device='cpu'):
        """
        Encode a list of text strings using BERT.
        texts: list[str]
        Returns: Tensor of shape [len(texts), bert_hidden_dim]
        """
        if len(texts) == 0:
            return torch.zeros((0, self.bert.config.hidden_size), device=device)

        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation
        cls_embeds = outputs.last_hidden_state[:, 0, :]
        return cls_embeds  # [batch_size, 768]

    def forward(self, node_texts, neighbors_texts, device='cpu'):
        """
        node_texts: list[str], the text for each node
        neighbors_texts: list[list[str]], each element is a list of neighbor texts for that node
        Returns: [num_nodes, num_classes]
        """
        # 1) Encode the node texts
        node_embeds = self.encode_texts(node_texts, device=device)  # [num_nodes, 768]

        # 2) Encode each node's neighbors
        neighbors_embeds_list = []
        for neigh_list in neighbors_texts:
            neigh_embeds = self.encode_texts(neigh_list, device=device)  # [num_neighbors, 768]
            neighbors_embeds_list.append(neigh_embeds)

        # 3) GNN aggregation
        x = self.gnn(node_embeds, neighbors_embeds_list)  # [num_nodes, gnn_hidden_dim]

        # 4) Classification
        logits = self.classifier(x)  # [num_nodes, 2]
        return logits


if __name__ == "__main__":
    # -----------------------------------------------
    # Example: We have 2 nodes, each with a text,
    # and each node has some neighbors' texts.
    node_texts = [
        "It is a sunny day, perfect for a walk in the park.",
        "Apple has just released a new iPhone model."
    ]

    neighbors_texts = [
        ["sunny day (neighbor 1)", "windy (neighbor 2)", "Central Park (neighbor 3)"],
        ["Apple Inc.", "Smartphones", "Tim Cook"]
    ]

    # Suppose we have binary labels for each node: 0 or 1
    # In a real task, 1 might mean "relevant" and 0 means "not relevant."
    node_labels = [1, 0]  # Just an example

    # Convert labels to a tensor (size = [num_nodes])
    node_labels_tensor = torch.tensor(node_labels, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = MyModel().to(device)

    # Define loss function (CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()
    # Define optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # We'll do a small number of epochs for demonstration
    num_epochs = 5

    model.train()  # set model to training mode
    for epoch in range(num_epochs):
        # Forward pass
        logits = model(node_texts, neighbors_texts, device=device)  # [num_nodes, 2]

        # Compute loss
        loss = criterion(logits, node_labels_tensor.to(device))  # node_labels_tensor shape: [num_nodes]

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training loss
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # After training, we can do inference in eval mode
    model.eval()
    with torch.no_grad():
        test_logits = model(node_texts, neighbors_texts, device=device)
        predictions = test_logits.argmax(dim=-1)
        print("\nInference:")
        print("Logits:", test_logits)
        print("Predicted labels:", predictions.cpu().numpy().tolist())
        print("Ground truth:", node_labels)
    
    
    save_path = "my_model_state.pt"

    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to {save_path}")