import solara
import pandas as pd
from typing import Optional, cast
import solara.express as solara_px

df = pd.read_csv('agent/data/averaged_full_state_data.csv')

state = solara.reactive(
    {
        'data': df ,
        'x':  solara.reactive('expected_tps'),
        'y':  solara.reactive('avg_num_request'),
        'logx': solara.reactive(False),
        'logy': solara.reactive(False),
        'size_max': solara.reactive(10.0),
        'size': solara.reactive('replica'),
        'color': solara.reactive('cpu_usage'),
        'filter': solara.reactive(None),
    }
    )

@solara.component
def FilteredDataFrame(df):
    filter = state.value['filter'].value
 
    dff = df
    if filter is not None:
        dff = df[filter]
    solara.DataFrame(dff, items_per_page=10)

@solara.component
def FilterPanel(df):
    solara.CrossFilterReport(df, classes=["py-2"])
    solara.CrossFilterSelect(df, configurable=False, column='replica')
    solara.CrossFilterSelect(df, configurable=False, column='cpu')
    solara.CrossFilterSelect(df, configurable=False, column='expected_tps')
    solara.CrossFilterSelect(df, configurable=False, column='previous_tps')



@solara.component
def ExecutionPanel():
    solara.Text("Execution Panel")

@solara.component
def DataViewer(df):
    input_cols, set_input_cols = solara.use_state(['replica'])

    filter = state.value['filter'].value

    dff = df
    if filter is not None:
        dff = df[filter]

    with solara.Sidebar():
        FilterPanel(df)
        with solara.Card("Controls", margin=0, elevation=0):
                with solara.Column():
                    columns = list(df.columns)
                    solara.SliderFloat(label="Size Max", value=state.value['size_max'], min=1, max=100, on_value=state.value['size_max'].set)
                    solara.Checkbox(label="Log x", value=state.value['logx'], on_value=state.value['logx'].set)
                    solara.Checkbox(label="Log y", value=state.value['logy'], on_value=state.value['logy'].set)
                    solara.Select("Size", values=columns, value=state.value['size'].value, on_value=state.value['size'].set)
                    solara.Select("Color", values=columns, value=state.value['color'].value, on_value=state.value['color'].set)
                    solara.Select("Column x", values=columns, value=state.value['x'].value, on_value=state.value['x'].set)
                    solara.Select("Column y", values=columns, value=state.value['y'].value, on_value=state.value['y'].set)

       
    solara.CrossFilterDataFrame(df, items_per_page=10)


    if state.value['x'].value and state.value['y'].value:
        solara_px.scatter(
            dff,
            state.value['x'].value,
            state.value['y'].value,
            size=state.value['size'].value,
            color=state.value['color'].value,
            size_max=state.value['size_max'].value,
            log_x=state.value['logx'].value,
            log_y=state.value['logy'].value,
        )
    else:
        solara.Warning("Select x and y columns")
            
@solara.component 
def Page():
    #if state.value['filter'].value is None:
    #    print('setting....')
    #    filter, set_filter = solara.use_cross_filter(id(state.value['data']))
    #    state.value['filter'].set(filter)

    DataViewer(state.value['data'])