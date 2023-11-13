import random
import re
import torch
import pandas as pd
import numpy as np

class Tokenizer(object):
    def __init__(self, vocab, unknown="[UNK]"):
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(vocab)}
        self.unknown = unknown
        if self.unknown not in self.token2idx:
            self.token2idx[self.unknown] = len(self.token2idx)

    def tokenize(self, text):
        return [token if token in self.token2idx else "[UNK]" for token in self._parse_text(text)]

    def convert_tokens_to_ids(self, tokens):
        return [self.token2idx.get(token, self.token2idx[self.unknown]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.idx2token.get(token_id, self.unknown) for token_id in ids]
    
    def _parse_text(self, text):
        raise NotImplementedError
    
    def random_token_id(self):
        return random.randint(0, len(self.idx2token))
    
class TextTokenizer(Tokenizer):
    def _parse_text(self, text):
        return text.split()
    
def load_csv(filepath):
    return pd.read_csv(filepath)

def to_sentence_df(text):
    sentences = re.sub("[.,;!?-]", '', text.lower()).split('\n') 
    sentences_df = pd.DataFrame([sentence for sentence in sentences if sentence], columns=['sequence'])
    vocab = [ '[PAD]', '[UNK]', '[MASK]' ] + list(set(" ".join(sentences).split()))
    return sentences_df, vocab

def generate_labeled_data(data, tokenizer, max_len, max_size=None):
    if max_size is not None:
        data = data[:max_size]
    labels = data['label'].values  # Extracts numpy array from DataFrame
    return _generate_masked_data(data, tokenizer, max_len, k=0, mask_rate=0.0, max_mask=0, noise_rate=0.0, max_size=max_size, dataset_size=None)[:1] + (torch.tensor(labels),)

def generate_masked_data(data, tokenizer, max_len, k, mask_rate=0.3, max_mask=3, noise_rate=0.0, max_size=None, dataset_size=None):
    return _generate_masked_data(data, tokenizer, max_len, k, mask_rate, max_mask, noise_rate, max_size, dataset_size)

def _generate_masked_data(data, tokenizer, max_len, k, mask_rate, max_mask, noise_rate, max_size, dataset_size):
    default_ignore_label = -100
    if max_size is not None:
        data = data[:max_size]

    if dataset_size is not None:
        data = _augment_sentence_df(data, dataset_size)

    input_ids = []
    segment_ids = []
    all_masked_lm_labels = []
    all_labels = []
    all_label_idxs = []

    attention_masks = []
    for _, row in data.iterrows():
        sequence = row['sequence']

        # Tokenize the sequence
        tokens = tokenizer.tokenize(sequence)
        ids = tokenizer.convert_tokens_to_ids(tokens)

        # Randomly insert noise tokens
        masked_seq = ids.copy()
        n_to_noise = int(len(tokens) * noise_rate)
        to_noise = random.sample(range(len(tokens)), n_to_noise)
        for pos in to_noise:
                masked_seq[pos] = tokenizer.random_token_id()
        
        # Mask mask_rate of tokens
        n_to_mask = max(min(max_mask, int(len(tokens) * mask_rate)), 1)
        to_mask = random.sample(range(len(tokens)), n_to_mask)
        masked_seq = ids.copy()
        masked_lm_labels =  [default_ignore_label] * max_len
        for pos in to_mask:
            for continuous_pos in range(k):
                next_masked_pos = pos + continuous_pos
                if next_masked_pos >= len(tokens):
                    next_masked_pos = pos - continuous_pos
                masked_seq[next_masked_pos] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
                masked_lm_labels[next_masked_pos] = ids[pos]

        # Zero Paddings
        attention_mask = [1] * len(masked_seq)
        if max_len > len(masked_seq):
            n_pad = max_len - len(masked_seq)
            masked_seq.extend([0] * n_pad)
            attention_mask.extend([0] * n_pad)

        # label Paddings
        labels = [ids[i] for i in to_mask]
        if max_mask > len(to_mask):
            n_pad = max_mask - len(to_mask)
            labels.extend([default_ignore_label] * n_pad)
            to_mask.extend([default_ignore_label] * n_pad)


        input_ids.append(masked_seq)
        segment_ids.append([0] * len(masked_seq)) # single-segment sequences
        all_masked_lm_labels.append(masked_lm_labels)
        all_labels.append(labels)
        all_label_idxs.append(to_mask)
        attention_masks.append(attention_mask)
    return torch.tensor(input_ids), torch.tensor(segment_ids), torch.tensor(all_masked_lm_labels), torch.tensor(all_label_idxs), torch.tensor(all_labels), torch.tensor(attention_masks)

def _augment_sentence_df(data, dataset_size):
    if dataset_size < len(data):
        print("Desired dataset_size is less than or equal to the original size.")
        return data
    
    # Calculate the number of rows to add
    n_rows_to_add = dataset_size - len(data)
    
    duplicate_indices = np.random.choice(data.index, size=n_rows_to_add, replace=True)
    data_expanded = pd.concat([data, data.loc[duplicate_indices]], ignore_index=True)
    return data_expanded