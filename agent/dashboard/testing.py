import solara
import torch
import solara.express as solara_px
from .training import local_state as training_state
from .data import state as data_state
from ..backend.utils import predict



local_state = solara.reactive(
    {
        'predictions': solara.reactive({}),
        'render_count': solara.reactive(0),
    }
    )

def force_render():
    local_state.value['render_count'].set(1 + local_state.value['render_count'].value)

@solara.component
def ScatterPlot(predictions, render_count):
    with solara.Card(title='Visual checks of the model performance',
                        subtitle='''Scatter plots of predicted and target outputs. More the dots
                        populate around the main diagonal (y=x), better the model performance.'''):
        for col in predictions.keys():
                with solara.Row():
                    for dataset in predictions[col].keys():
                        solara_px.scatter(
                            predictions[col][dataset],
                            x = 'prediction',
                            y = 'target',
                            title=f'{col} - {dataset}'
                    )
            

@solara.component
def Page():
    df = data_state.value['data']

    filter, set_filter = solara.use_cross_filter(id(df))

    dff = df
    if filter is not None:
        dff = df[filter]

    def make_predictions():
        model = training_state.value['model'].value
        if model is None:
            print('There is no pre-trained model! Please train your model.')
        else:
            print('There is a pre-trained model')
            input_cols = training_state.value['input_cols'].value
            output_cols = training_state.value['output_cols'].value
            trn_ratio = training_state.value['trn_ratio'].value
            batch_size_trn = training_state.value['batch_size_trn'].value
            batch_size_val = training_state.value['batch_size_val'].value
            seed = training_state.value['seed'].value
            predictions = predict(model, dff, input_cols, output_cols, trn_ratio, 
                batch_size_trn, batch_size_val, seed)
        
            local_state.value['predictions'].set(predictions)
            force_render()
            
    solara.Button(label='Output Predictions', on_click=make_predictions)
    ScatterPlot(local_state.value['predictions'].value, local_state.value['render_count'].value)
