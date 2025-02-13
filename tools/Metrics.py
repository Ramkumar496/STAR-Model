import torch

def get_recall(indices, targets): 
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
    hits_count = torch.sum(hits).item()
    total = targets.size(0)

    recall = hits_count / total if total > 0 else 0.0
    return hits, recall

def get_precision(indices, targets):
    """
    Calculates the precision score for the given predictions and targets.

    Args:
        indices (torch.Tensor): Tensor of shape [batch_size, k] containing top-k indices.
        targets (torch.Tensor): Tensor of shape [batch_size] containing ground truth indices.

    Returns:
        float: Precision score.
    """
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
    hits_count = torch.sum(hits).item()
    total_predictions = indices.numel()

    precision = hits_count / total_predictions if total_predictions > 0 else 0.0
    return precision

def get_accuracy(logits, targets):
    """
    Calculates the accuracy score for the given logits and targets.

    Args:
        logits (torch.Tensor): Tensor of shape [batch_size, n_classes] containing predicted logits.
        targets (torch.Tensor): Tensor of shape [batch_size] containing ground truth indices.

    Returns:
        float: Accuracy score.
    """
    # Get the predicted class (argmax along the last dimension)
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def get_mrr(indices, targets):
    """
    Calculates the Mean Reciprocal Rank (MRR@K) score for the given predictions and targets.
    Assumes indices are top-k predictions for each sample.

    Args:
        indices (torch.Tensor): Tensor of shape [batch_size, k] containing top-k indices.
        targets (torch.Tensor): Tensor of shape [batch_size] containing ground truth indices.

    Returns:
        float: MRR@K score.
    """
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero(as_tuple=False)

    if hits.size(0) == 0:  # No hits, return 0 for MRR
        return 0.0

    ranks = hits[:, 1].float() + 1  # Extract ranks (1-based index from dim=1)
    rranks = 1.0 / ranks  # Reciprocal of ranks
    mrr = rranks.mean().item()
    return mrr

def calc(logits, targets, k=20):
    """
    Compute Recall@K, MRR@K, Precision@K, and Accuracy.

    Args:
        logits (torch.Tensor): Tensor of shape [batch_size, n_classes] containing predicted logits.
        targets (torch.Tensor): Tensor of shape [batch_size] containing ground truth indices.
        k (int): The number of top predictions to consider.

    Returns:
        dict: Dictionary containing Recall@K, MRR@K, Precision@K, and Accuracy.
    """
    # Get the top-k indices from logits
    _, indices = torch.topk(logits, k, dim=-1)

    # Compute metrics
    _, recall = get_recall(indices, targets)
    precision = get_precision(indices, targets)
    mrr = get_mrr(indices, targets)
    accuracy = get_accuracy(logits, targets)

    return {
        "Recall@K": recall,
        "Precision@K": precision,
        "MRR@K": mrr,
        "Accuracy": accuracy
    }
