import solara
import pandas as pd
from typing import Optional, cast
import solara.express as solara_px
import numpy as np

def read_data():
    df = pd.read_csv('agent/data/averaged_full_state_data.csv')
    df = df.infer_objects()
    df['step'] = df.index
    for col in df.columns:
        if df.dtypes[col] == np.float64:
            df[col] = df[col].apply(lambda x: round(x, 6))
    return df

df = read_data()

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
    with solara.Column(gap="0px"):
        solara.CrossFilterReport(df)
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
        with solara.Card("Cross Filters", margin=1, elevation=10,
            subtitle="""You can filter the data by selecting different values of several attributes,
            and investigate the characteristics of the data.
            Once a filter is applied to an attribute, other filter boxes
            are immediately updated as well as the number of records in the
            filtered data."""):
            FilterPanel(df)
        with solara.Card("Plot Settings", margin=1, elevation=10,
                subtitle="""You can adjust these parameters to control the 2D scatter plot on the right.
                """):
                with solara.Column(gap="0px"):
                    columns = list(df.columns)
                    with solara.Row():
                        with solara.Tooltip("Select to draw the x-axis in logarithmic scale"):
                            solara.Checkbox(label="Log x", value=state.value['logx'], on_value=state.value['logx'].set)
                        with solara.Tooltip("Select to draw the y-axis in logarithmic scale"):
                            solara.Checkbox(label="Log y", value=state.value['logy'], on_value=state.value['logy'].set)
                    with solara.Tooltip("Maximum size of the markers in the scatter plot"):
                        solara.SliderFloat(label="Maximum Marker Size", value=state.value['size_max'], min=1, max=100, on_value=state.value['size_max'].set)
                    with solara.Tooltip("Adjust the marker size based on the selected attribute"):
                        solara.Select("Size", values=columns, value=state.value['size'].value, on_value=state.value['size'].set)
                    with solara.Tooltip("Adjust the marker color based on the selected attribute"):
                        solara.Select("Color", values=columns, value=state.value['color'].value, on_value=state.value['color'].set)
                    with solara.Tooltip("Select the attribute to be used in x-axis"):
                        solara.Select("X-Axis", values=columns, value=state.value['x'].value, on_value=state.value['x'].set)
                    with solara.Tooltip("Select the attribute to be used in y-axis"):
                        solara.Select("Y-Axis", values=columns, value=state.value['y'].value, on_value=state.value['y'].set)

    with solara.Card("Raw Data", margin=1, elevation=2,
        subtitle="""The raw data to be used in this study.
        Cross-filters on the left are immediately applied to the data.
        Data is visualized as a scatter plot below which can be controlled via
        widgets in 'Plot Settings'. The filters and the visualization plot
        are only used to understand the characteristics of the data. So, the filtered data
        is not used in the next steps. In the training section, there will be other
        filters specific to training.
        """):
        solara.CrossFilterDataFrame(df, items_per_page=10)

    with solara.Card("Scatter Plot", margin=1, elevation=2,
        subtitle="""Based on the plot controls on the left, the
        data is visualized as a 2D scatter plot. You can select different
        attributes for X and Y-axis. By manipulating the size and color
        properties of the markers, you can investigate the data in detail.
        """):

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
                width=800,
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