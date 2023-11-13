import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import BertTokenizer, BertConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import BertTokenizer, BertConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from types import SimpleNamespace

from utils import visualization
from utils import models

import math
import random
import numpy as np

# ChatGPT generated text data about transformers.
def get():
    return (
        "Transformers change how we deal with language.\n"
        "They help machines understand text.\n"
        "They read words and see patterns.\n"
        "Patterns help them learn.\n"
        "Learning like this was hard before.\n"
        "Now, it's much easier because of transformers.\n"
        "Transformers came from a need.\n"
        "This need was understanding language better.\n"
        "Before transformers, we had other models.\n"
        "These models were good but not great.\n"
        "They worked in a simple way.\n"
        "They read text one word at a time.\n"
        "But language isn't that simple.\n"
        "It's complex.\n"
        "Imagine reading a book.\n"
        "You don't read just one word, you read many words.\n"
        "Words link with other words.\n"
        "They form ideas when they link.\n"
        "The old models missed these links.\n"
        "They didn't see the big picture.\n"
        "Transformers were built to see these links.\n"
        "A man named Vaswani made transformers.\n"
        "He had a team.\n"
        "The team was smart.\n"
        "They asked, how can we read words in a better way?\n"
        "They thought hard.\n"
        "They used math and computers.\n"
        "They created the transformer model.\n"
        "The model had new ideas.\n"
        "One idea was attention.\n"
        "Attention is a smart way to read words.\n"
        "It makes words look at other words.\n"
        "They don't just look at one; they look at many.\n"
        "This way, they form better ideas.\n"
        "For example, the word bank.\n"
        "It means a place for money.\n"
        "But it can also mean the side of a river.\n"
        "How do you know the right meaning?\n"
        "The old models didn't know.\n"
        "Transformers can know.\n"
        "They see the words around bank.\n"
        "They use attention to understand.\n"
        "But attention wasn't easy.\n"
        "It needed lots of power.\n"
        "Computers had to work hard.\n"
        "They saw all words at once.\n"
        "They needed to remember lots.\n"
        "This remembering is called memory.\n"
        "Transformers use lots of memory.\n"
        "Memory was a problem.\n"
        "The team had to solve it.\n"
        "They used special tricks.\n"
        "These tricks made transformers better.\n"
        "Now, they could work with big texts.\n"
        "They could learn from lots of words.\n"
        "This learning is important.\n"
        "When transformers learn, they see patterns.\n"
        "Sometimes, these patterns are hard to see.\n"
        "People can't see them, but transformers can.\n"
        "They find meaning in the patterns.\n"
        "This meaning helps them understand language.\n"
        "Understanding language is key.\n"
        "It lets transformers do many tasks.\n"
        "They can write like a person.\n"
        "They can read and answer questions.\n"
        "They can talk to people.\n"
        "All these tasks are because they understand words.\n"
        "People saw transformers were good.\n"
        "They started using them more.\n"
        "Big companies said, we will use transformers.\n"
        "Schools said, we will study transformers.\n"
        "Everyone wanted to be part of the work.\n"
        "Now, many things changed.\n"
        "Machines became smarter.\n"
        "They helped people more.\n"
        "They could do new jobs.\n"
        "Jobs that were hard are now easy.\n"
        "It's all because of transformers.\n"
        "They made a big difference.\n"
        "Transformers keep getting better.\n"
        "People find new ways to use them.\n"
        "They help in schools, hospitals, and more.\n"
        "They make life easier.\n"
        "They save time.\n"
        "They are a big step for machines.\n"
        "We see how important transformers are.\n"
        "They changed how machines learn.\n"
        "They made them smart.\n"
        "They gave them a way to understand us.\n"
        "They use attention and memory in smart ways.\n"
        "They solve problems in language.\n"
        "The future is bright for transformers.\n"
        "They will help us in many ways.\n"
        "We will keep learning from them.\n"
        "They will keep getting better.\n"
        "One day, they might surprise us all.\n"
        "It's a journey of discovery with transformers.\n"
    )

def train_wikitext(device, positional_embedding, attention):
    train_wikitext_subset = load_dataset('wikitext', 'wikitext-2-v1', split=f'train')
    train_text_data = train_wikitext_subset['text']

    validation_wikitext_subset = load_dataset('wikitext', 'wikitext-2-v1', split=f'validation')
    validation_text_data = validation_wikitext_subset['text']

    text_max_len = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=text_max_len)

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=text_max_len):
            self.tokenizer = tokenizer
            self.input_ids = []
            self.attention_masks = []
            for text in texts:
                encoded_text = tokenizer(text, max_length=text_max_len, padding='max_length', truncation=True, return_tensors='pt')
                self.input_ids.append(encoded_text.input_ids)
                self.attention_masks.append(encoded_text.attention_mask)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {'input_ids': self.input_ids[idx].squeeze(), 'attention_mask': self.attention_masks[idx].squeeze()}

    train_dataset = TextDataset(train_text_data, tokenizer)
    validation_dataset = TextDataset(validation_text_data, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data_collator)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False, collate_fn=data_collator)

    text_config = SimpleNamespace(
            vocab_size=tokenizer.vocab_size,
            hidden_size=60,
            max_position_embeddings=text_max_len,
            type_vocab_size=1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            num_attention_heads=6,
            hidden_act="gelu",
            intermediate_size=120,
            num_hidden_layers=1,
            is_decoder=False,
            output_attentions=True,
            output_hidden_states=False,
            pruned_heads = {},
            initializer_range=0.02,
            device=device
        )

    model = models.BertForMaskedLM(config=text_config, positional_embedding=positional_embedding, attention=attention)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print("Start to train model on wikitext...")
    for epoch in range(50):
        total_train_loss, total_masked_tokens, correct_masked_predictions = 0.0, 0, 0
        model.train()

        for batch in train_dataloader:
            optimizer.zero_grad() ## Don't set gradients to zero before gradient update step

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss, outputs, attentions = model(
            input_ids=input_ids,
            masked_lm_labels=labels,
            attention_mask=attention_mask
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            masked_positions = (labels != -100)
            total_masked_tokens += torch.sum(masked_positions)
            predicted_tokens = torch.argmax(outputs, dim=-1)
            correct_masked_predictions += torch.sum(predicted_tokens[masked_positions] == labels[masked_positions])

        if (epoch + 1) % 1 == 0:
            train_accuracy = correct_masked_predictions.float() / total_masked_tokens.float()
            avg_train_loss = total_train_loss / len(train_dataloader)

            model.eval()
            total_eval_loss, total_masked_tokens, correct_masked_predictions = 0.0, 0, 0
            for batch in validation_dataloader:
                with torch.no_grad():
                  input_ids = batch['input_ids'].to(device)
                  attention_mask = batch['attention_mask'].to(device)
                  labels = batch['labels'].to(device)

                  test_loss, outputs, attentions = model(
                      input_ids=input_ids,
                      masked_lm_labels=labels,
                      attention_mask=attention_mask
                  )
                  total_eval_loss += loss.item()

                  masked_positions = (labels != -100)
                  total_masked_tokens += torch.sum(masked_positions)
                  predicted_tokens = torch.argmax(outputs, dim=-1)
                  correct_masked_predictions += torch.sum(predicted_tokens[masked_positions] == labels[masked_positions])
            avg_eval_loss = total_eval_loss / len(validation_dataloader)
            eval_accuracy = correct_masked_predictions.float() / total_masked_tokens.float()
            print('Epoch:', '%04d' % (epoch + 1), 'train loss =', '{:.6f}'.format(avg_train_loss), 'train acc =', '{:.6f}'.format(train_accuracy), 'test loss =', '{:.6f}'.format(avg_eval_loss), 'test acc =', '{:.6f}'.format(eval_accuracy))