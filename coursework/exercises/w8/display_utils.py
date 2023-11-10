
import ipywidgets as widgets
from IPython.display import display, clear_output


global n_way
n_way = 5
global n_support
n_support = 5
global n_query
n_query = 15
global n_train_episode 
n_train_episode = 5

def sliders(parameters):
    n_way_slider = widgets.IntSlider(value=n_way, min=1, max=10, description='n_way:', style = {'description_width': 'initial'})
    n_support_slider = widgets.IntSlider(value=n_support, min=1, max=10, description='n_support:', style = {'description_width': 'initial'})
    n_query_slider = widgets.IntSlider(value=n_query, min=1, max=10, description='n_query:', style = {'description_width': 'initial'})
    n_train_episode_slider = widgets.IntSlider(value=n_train_episode, min=0, max=100, step=5, description='n_train_episode:', style = {'description_width': 'initial'})

    def on_value_change(change):
        parameters['n_way'] = n_way_slider.value
        parameters['n_support'] = n_support_slider.value
        parameters['n_query'] = n_query_slider.value
        parameters['n_train_episode'] = n_train_episode_slider.value
        clear_output(wait=True)


    n_way_slider.observe(on_value_change, names="value")
    n_support_slider.observe(on_value_change, names="value")
    n_query_slider.observe(on_value_change, names="value")
    n_train_episode_slider.observe(on_value_change, names="value")

    display(widgets.HBox([n_way_slider, n_support_slider, n_query_slider, n_train_episode_slider]))