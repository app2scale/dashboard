import solara
import pandas as pd
from typing import Optional, cast
import solara.express as solara_px
from .data import state
from ..backend.data import ExplorationDataset
from ..backend.utils import train
from ..backend.loss import loss_mape

local_state = solara.reactive(
    {
        'input_cols': solara.reactive(['replica','cpu','expected_tps','previous_tps']),
        'output_cols': solara.reactive(['cpu_usage','num_request']),
        'trn_ratio' : solara.reactive(0.8),
        'learning_rate_log10': solara.reactive(-3),
        'batch_size_trn': solara.reactive(32),
        'batch_size_val': solara.reactive(16),
        'model_name': solara.reactive("Perceptron"),
        'optimizer_name': solara.reactive("Adam"),
        'max_epoch': solara.reactive(30),
        'loss_name': solara.reactive('mae'),
        'loss_plot_data': solara.reactive({'epoch': [], 'trn_loss': [], 'val_loss': []}),
        'render_count': solara.reactive(0),
        'model': solara.reactive(None),
        'ds': solara.reactive(None),
        'seed': solara.reactive(42),
    }
    )

@solara.component
def LossPlot(data, render_count, log_y):
    trn_loss_data = [[e, l] for e, l in zip(data['epoch'], data['trn_loss'])]
    val_loss_data = [[e, l] for e, l in zip(data['epoch'], data['val_loss'])]
    options = {
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross'
            }
        },
        "xAxis": {
            "type": "value",
            "name": "epoch",
        },
        "yAxis": {
            "type": "log" if log_y else "value",
            "name": "loss",
        },
        "series": [ 
            {
                "name": "training loss",
                "data": trn_loss_data,
                "type": 'line'
            },
            {
                "name": "validation loss",
                "data": val_loss_data,
                "type": 'line'
            },            
        ]
    }

    with solara.Column():
        solara.FigureEcharts(option=options)

def force_render():
    local_state.value['render_count'].set(1 + local_state.value['render_count'].value)

@solara.component
def ExecutePanel(df):
    filter, set_filter = solara.use_cross_filter(id(df))
    loss_plot_log_y, set_loss_plot_log_y  = solara.use_state_or_update(False)

    dff = df
    if filter is not None:
        dff = df[filter]

    def trigger_training():

        input_cols = local_state.value['input_cols'].value
        output_cols = local_state.value['output_cols'].value
        trn_ratio = local_state.value['trn_ratio'].value
        batch_size_trn = local_state.value['batch_size_trn'].value
        batch_size_val = local_state.value['batch_size_val'].value
        learning_rate_log10 = local_state.value['learning_rate_log10'].value
        learning_rate = 10**learning_rate_log10
        optimizer_name = local_state.value['optimizer_name'].value
        max_epoch = local_state.value['max_epoch'].value
        loss_name = local_state.value['loss_name'].value
        seed = local_state.value['seed'].value
        model_name = local_state.value['model_name'].value

        epoch_list = []
        trn_loss_list = []
        val_loss_list = []
        ds = ExplorationDataset(dff, input_cols, output_cols)
        local_state.value['ds'].set(ds)
        for epoch, trn_loss, val_loss, model in train(ds, model_name, trn_ratio,
              batch_size_trn, batch_size_val, optimizer_name, learning_rate,
              max_epoch, loss_name, seed):
            epoch_list.append(epoch)
            trn_loss_list.append(trn_loss)
            val_loss_list.append(val_loss)
            local_state.value['loss_plot_data'].set(
                {'epoch':epoch_list,
                 'trn_loss': trn_loss_list,
                 'val_loss': val_loss_list})
            force_render()
            local_state.value['model'].set(model)
    solara.Button(label='Train', on_click=trigger_training)
    with solara.Card(title="Loss History",  margin=1, elevation=10,
                    subtitle="""Once you start training, you can monitor the training and validation losses in this plot.
                    """):
        solara.Checkbox(label='Log-Scale y-axis', value=loss_plot_log_y, on_value=set_loss_plot_log_y)
        LossPlot(local_state.value['loss_plot_data'].value, 
                 local_state.value['render_count'].value, 
                 loss_plot_log_y)


@solara.component
def ParameterSelection(df, attributes):
    def select_input_cols(selected_cols):
        local_state.value['input_cols'].set(selected_cols)
    def select_output_cols(selected_cols):
        local_state.value['output_cols'].set(selected_cols)

    with solara.lab.Tabs():
        with solara.lab.Tab("I/O"):
            with solara.Card(title="Input/Output Selection",  margin=1, elevation=10,
                subtitle="""Select which attributes of the data are to be used as input/output of the
                machine learning model. Selected attributes will be reflected to the dataframe on the right,
                immediately."""):
                solara.SelectMultiple(label='Input features', all_values=attributes,
                            values=local_state.value['input_cols'].value,
                            on_value=select_input_cols)
                solara.SelectMultiple(label='Output features', all_values=attributes,
                            values=local_state.value['output_cols'].value,
                            on_value=select_output_cols)
        with solara.lab.Tab("PARAMETERS"):
            with solara.Card(title="Model Training Parameters",  margin=1, elevation=10,
                subtitle="""Select the machine learning model, optimizer, loss and various hyper-parameters
                used in the training."""):
                solara.Select(label="Model", values=["Perceptron","NetSingleHiddenLayer"],
                            value=local_state.value['model_name'].value,
                            on_value=local_state.value['model_name'].set)
                solara.Select(label="Optimizer", values=["Adam"],
                            value=local_state.value['optimizer_name'].value,
                            on_value=local_state.value['optimizer_name'].set)
                solara.Select(label="Loss", values=['mape','mae'], 
                            value=local_state.value['loss_name'].value,
                            on_value=local_state.value['loss_name'].set)
                solara.SliderFloat(label="Learning rate (log10)",
                                value=local_state.value['learning_rate_log10'].value,
                                min=-4, max=1, step=0.01,
                                on_value=local_state.value['learning_rate_log10'].set)
                solara.SliderInt(label='Max epoch',
                                value=local_state.value['max_epoch'].value, min=1, max=1000,
                                on_value=local_state.value['max_epoch'].set,
                                thumb_label=True)
                solara.SliderFloat(label='Training ratio', 
                                value=local_state.value['trn_ratio'].value, min=0, max=1,
                                on_value=local_state.value['trn_ratio'].set,
                                thumb_label=True)
                solara.SliderInt(label='Batch size training', 
                                value=local_state.value['batch_size_trn'].value, min=1, max=256,
                                on_value=local_state.value['batch_size_trn'].set,
                                thumb_label=True)   
                solara.SliderInt(label='Batch size validation', 
                                value=local_state.value['batch_size_val'].value, min=1, max=256,
                                on_value=local_state.value['batch_size_val'].set,
                                thumb_label=True)  
                solara.SliderInt(label='random seed', value=local_state.value['seed'].value, min=0, max=1000,
                                on_value=local_state.value['seed'].set,
                                thumb_label=True)
        with solara.lab.Tab("FILTER"):
            with solara.Card(title="Data Filter",  margin=1, elevation=10,
                    subtitle="""In addition to the input/output attributes, you can also
                    select a subset of rows for training. Filtered dataframe displayed on the
                    right  will be used in the training."""):
                solara.CrossFilterReport(df)
                for col in ['replica','cpu','expected_tps','previous_tps']:
                    if col in df.columns:
                        solara.CrossFilterSelect(df, configurable=False, column=col)


@solara.component
def Page():
    df = state.value['data']
    attributes = list(df.columns)
    dff = df
    filtered_cols = []
    if len(local_state.value['input_cols'].value) > 0:
        filtered_cols += local_state.value['input_cols'].value
    if len(local_state.value['output_cols'].value) > 0:
        filtered_cols += local_state.value['output_cols'].value
    if len(filtered_cols) > 0:
        dff = df[filtered_cols]
    with solara.Sidebar():
        ParameterSelection(dff, attributes)
    with solara.Row():
        with solara.Card(title="Training/Testing Data",  margin=1, elevation=10,
                    subtitle="""Based on the selected nput/output attributes and the cross-filters,
                    this is the final data used in training/validation. Right before using
                    this data in the training/validation, normalization is applied. For more
                    information please check backend.data.ExplorationDataset""",
                    style='width: 500px'):
            solara.CrossFilterDataFrame(dff, items_per_page=10)
        ExecutePanel(dff)

    