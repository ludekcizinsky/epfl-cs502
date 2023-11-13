import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
import numpy as np

def display_attantion(attentions, tokenizer, input_ids, layer=0,  head=0):

    def visualize_attention(sample_id):
        attention = attentions[layer][sample_id, head].detach().numpy()
        plt.figure(figsize=(2.5, 2.5))
        tick_labels = tokenizer.convert_ids_to_tokens(input_ids[sample_id,:].detach().to("cpu").numpy())
        sns.heatmap(attention, cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels)
        plt.title(f"Attention Weights - Layer {layer+1}, Head {head+1}, Sample {sample_id}")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.show()

    num_samples = attentions[layer].shape[0]
    options = np.random.choice(range(num_samples), size=20, replace=False)
    options.sort()
    sample_idx_selector = widgets.Dropdown(
        options=options,
        value=options[0],
        description='Sample:',
    )
    return visualize_attention, sample_idx_selector


def display_multi_attantion(attentions, tokenizer, input_ids, layers,  heads):

    def visualize_attention(sample_id, layer, head):
        def transform_ticks(tick_labels, input_ids, sample_id):
            labels = tokenizer.convert_ids_to_tokens(input_ids[sample_id, tick_labels].detach().to("cpu").numpy())
            return [f'{v} ({i})' for v, i in zip(labels, [int(i) for i in tick_labels])]

        attention = attentions[layer-1][sample_id, head-1].detach().to("cpu").numpy()
        plt.figure(figsize=(2.5, 2.5))
        ax = sns.heatmap(attention, cmap="Blues")
        xticks_labels = transform_ticks(ax.get_xticks(), input_ids, sample_id)
        yticks_labels = transform_ticks(ax.get_yticks(), input_ids, sample_id)

        ax.set_xticklabels(xticks_labels, rotation=90)
        ax.set_yticklabels(yticks_labels)

        plt.title(f"Attention Weights \n Layer {layer}, Head {head}, Sample {sample_id}")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.show()

    num_samples = attentions[0].shape[0]
    options = np.random.choice(range(num_samples), size=20, replace=False)
    options.sort()
    sample_idx_selector = widgets.Dropdown(
        options=options,
        value=options[0],
        description='Sample:',
    )

    layer_selector = widgets.Dropdown(
        options=layers,
        value=layers[0],
        description='Layer:',
    )

    head_selector = widgets.Dropdown(
        options=heads,
        value=heads[-1],
        description='Head:',
    )
    return visualize_attention, sample_idx_selector, layer_selector, head_selector


def display_positional_encoding(pos_embedding_func):

    def visualize_positional_embeddings(max_len, dimension):
        pos_emb = pos_embedding_func(max_len, dimension, "cpu").embedding()
        plt.figure(figsize=(8, 2.5))
        plt.pcolormesh(pos_emb.squeeze(0), cmap='viridis')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Token Position')
        plt.title('Positional Encoding')
        plt.colorbar(label='Embedding Value')
        plt.show()

    dimension_selector = widgets.Dropdown(
        options=[10, 15, 20, 30, 50, 100, 150, 300],
        value=100,
        description='Hidden Dimension:',
        style = {'description_width': 'initial'}
    )

    max_len_selector = widgets.Dropdown(
        options=[100, 500, 1000, 5000],
        value=1000,
        description='Sequence Length',
        style = {'description_width': 'initial'}
    )
    return visualize_positional_embeddings, dimension_selector, max_len_selector



