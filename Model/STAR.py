import torch
from torch import nn
from tools import Metrics
from tools import Optimizer
from model.Modules import Module_1_6, Module_2, Module_3, Module_4, Module_5
from torch.nn import LSTM
import torch.nn.functional as F
from tools.Metrics import get_recall, get_mrr


class STAR(nn.Module):
    def __init__(self, device, n_items, args):
        super(STAR, self).__init__()

        self.device = device
        self.module_1_6 = Module_1_6(self.device, n_items, args.embedding_dim, args.dropout)
        self.module_2 = Module_2(self.device, args.embedding_dim, args.hidden_size, args.dropout)
        self.module_3 = Module_3(self.device, args.embedding_dim, args.dropout)
        self.module_4 = Module_4(self.device, args.dropout)
        self.module_5 = Module_5(self.device, args.embedding_dim, args.hidden_size, dropout=args.dropout, noise_epsilon=args.epsilon)
        self.lstm = LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1, batch_first=True)
        self.epsilon = args.epsilon  # epsilon value for Gaussian noise

        # Add a fully connected layer to ensure output shape is (1, 512)
        self.fc_out = nn.Linear(args.hidden_size, 512)

        # Move model to the specified device
        self.to(self.device)

    def forward(self, item_sequence, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o):
        embeddings = self.module_1_6(item_sequence, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o)
        embeddings[0] = embeddings[0].view(1, -1, 180)

        hidden_states = self.module_2(embeddings[0])

        b_embedding, a_embedding = self.module_3(embeddings[1], embeddings[2], embeddings[3], embeddings[4], embeddings[5], embeddings[6])

        weighted_gru_out, weighted_a_embedding, gru_out, b_embedding, a_embedding = self.module_4(hidden_states, b_embedding, a_embedding)

        weighted_features = [weighted_gru_out, weighted_a_embedding]
        max_size = max(tensor.size(1) for tensor in weighted_features)
        weighted_features = [torch.cat((tensor, torch.zeros(tensor.size(0), max_size - tensor.size(1), tensor.size(2)).to(tensor.device)), dim=1) for tensor in weighted_features]
        concatenated_inputs = torch.cat(weighted_features, dim=-1)

        concatenated_inputs = concatenated_inputs.view(concatenated_inputs.size(0), -1, 180)

        final_hidden = self.module_5(concatenated_inputs)  # Output shape: [batch_size, 512]

        # Apply Gaussian noise
        noise = torch.randn_like(final_hidden) * self.epsilon
        noise = noise.to(final_hidden.device)
        final_hidden += noise

        lstm_output, _ = self.lstm(concatenated_inputs)

        output = lstm_output[:, -1, :]  # Get output for the last time step

        # Apply fully connected layer to ensure output size is 512
        output_512 = self.fc_out(output)

        # Here, make sure Y_hat has the correct batch size
        Y_tilda = self.module_1_6.get_scores(output_512)

        # Expand Y_tilda to match the batch size of target (512)
        batch_size = item_sequence.size(0)  # Ensure batch size is taken from item_sequence
        Y_tilda = Y_tilda.expand(batch_size, -1)

        return Y_tilda


    def set_initial_weights(self, initial_weights):
        self.module_1_6.set_initial_embedds(initial_weights)


class STARFramework(nn.Module):
    def __init__(self, device, n_items, args):
        super(STARFramework, self).__init__()
        self.star = STAR(device, n_items, args)  # The actual model
        self.device = device
        self.n_items = n_items
        self.loss_function = nn.CrossEntropyLoss()
        self.set_optimizer(args)

    def set_optimizer(self, args):
        self.optimizer = Optimizer.Optimizer(
            filter(lambda p: p.requires_grad, self.star.parameters()), args
        )

    def set_initial_weights(self, FE_initial_weights):
        self.star.set_initial_weights(FE_initial_weights)

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step_scheduler(self):
        self.optimizer.step_scheduler()

    def forward(self, item_sequence, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o):
        return self.star(item_sequence, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o)

    import torch.nn.functional as F

    def calc_loss(self, Y_hat, target):
        # Ensure target is a long tensor and on the correct device
        target = target.clone().detach().long().to(self.device)

        # Debugging: Check shapes before proceeding
        #print(f"Y_hat shape in loss: {Y_hat.shape}")
        #print(f"Target shape in loss: {target.shape}")

        # Check if target values are within the valid range of class indices
        target_max = target.max().item()
        target_min = target.min().item()
        num_classes = Y_hat.size(1)  # Number of classes (512)

        #print(f"Target min: {target_min}, Target max: {target_max}")
        #print(f"Number of classes: {num_classes}")

        # If target values are out of bounds, clip them to the valid range [0, num_classes-1]
        if target_max >= num_classes:
            #print("Warning: Target values out of range, clipping them.")
            target = torch.clamp(target, 0, num_classes - 1)

        # Ensure Y_hat is not scalar
        if Y_hat.dim() == 0:
            print("Warning: Y_hat is a scalar, reshaping it.")
            Y_hat = Y_hat.unsqueeze(0)  # Add batch dimension if Y_hat is scalar

        # Check for dimensional consistency
        assert Y_hat.dim() == 2, f"Y_hat should have 2 dimensions, but has {Y_hat.dim()} dimensions"
        assert target.dim() == 1, f"Target should have 1 dimension, but has {target.dim()} dimensions"
        assert Y_hat.size(0) == target.size(0), f"Batch size mismatch: Y_hat batch size {Y_hat.size(0)}, target batch size {target.size(0)}."

        # Compute and return the cross entropy loss
        loss = F.cross_entropy(Y_hat, target)
        print(f"Loss: {loss.item()}")
        return loss




    def fit(self, item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o):
        # Forward pass to get predictions (Y_hat)
        Y_hat = self.forward(item_sequence, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o)

        # Ensure target is a long tensor and on the correct device
        target = target.clone().detach().long().to(self.device)

        # Flatten target to ensure it's of shape [512] (assuming batch size 512)
        target = target.view(-1)  # This flattens target to [512], making it compatible for cross_entropy
        
        # Ensure target has the correct shape
        if target.dim() == 0:  # If target becomes a scalar, you should reshape it or handle it
            target = target.unsqueeze(0)  # Add a dimension if target is a scalar

        # Ensure target has the correct batch size and check dimension
        assert target.dim() == 1, f"Expected target to be 1D, but got {target.shape}"
        assert target.size(0) == Y_hat.size(0), f"Mismatch: target batch size {target.size(0)} does not match Y_hat batch size {Y_hat.size(0)}."

        # If Y_hat has batch size 1, we need to expand it to match the batch size of target
        if Y_hat.size(0) != target.size(0):
            Y_hat = Y_hat.expand(target.size(0), -1)  # Expand Y_hat to match target batch size

        # Proceed with loss calculation
        return self.calc_loss(Y_hat, target)


    

    def test(self, item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o):
        # Forward pass to compute Y_hat
        Y_hat = self.forward(item_sequence, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o)
        
        # Debugging the shapes of Y_hat and target
        print(f"Y_hat shape: {Y_hat.shape}")  # Debug
        print(f"Target shape: {target.shape}")  # Debug
        
        # Ensure target has the correct shape [batch_size] (1D tensor)
        if target.dim() > 1:
            target = target.view(-1)  # Flatten target to a 1D tensor
            print(f"Target shape after reshape: {target.shape}")  # Debug

        # Proceed with loss calculation
        loss = self.calc_loss(Y_hat, target)
        print(f"Calculated loss: {loss.item():.4f}")  # Debug loss value

        # Debugging the top-k indices of predictions for recall/MRR computation
        top_k = 20  # Adjust the value of k if needed
        _, top_indices = torch.topk(Y_hat, top_k, dim=-1)
        #print(f"Top-{top_k} indices shape: {top_indices.shape}")
        
        # Debug the target shape before passing to metrics functions
        #print(f"Target shape for metrics: {target.shape}")

        # Use metrics from the metrics.py file
        hits, recall = get_recall(top_indices, target)
       

        mrr = get_mrr(top_indices, target)
        

        return loss, recall, mrr



