import random
import torch

def masked_label_accuracy(labels, labels_idx, outputs):
    acc_masked = 0
    total_count = 0

    tokens_predictions = outputs.max(2)[1]
    for batch in range(len(labels)):
        token_idxs_mask, token_labels_mask = labels_idx[batch] >= 0, labels[batch]  >= 0
        token_idxs, token_labels = labels_idx[batch][token_idxs_mask], labels[batch][token_labels_mask]
        tokens_prediction = tokens_predictions[batch][token_idxs]
        acc_masked += sum(tokens_prediction == token_labels).item()
        total_count += len(token_labels)
    return acc_masked / total_count

def model_masked_label_accuracy(model, data_loader, device):
    with torch.no_grad():
        acc_masked = 0
        for batch_input_ids, batch_segment_ids, batch_masked_lm_labels, batch_masked_pos, batch_masked_tokens, _ in data_loader:
            _, outputs, _, attentions = model(
            input_ids=batch_input_ids.to(device),
            token_type_ids=batch_segment_ids.to(device),
            masked_lm_labels=batch_masked_lm_labels.to(device),
            )
            local_acc = masked_label_accuracy(batch_masked_tokens, batch_masked_pos, outputs.data.detach().to("cpu"))
            acc_masked += local_acc
        average_acc = acc_masked / len(data_loader)
    return average_acc, attentions