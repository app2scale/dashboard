import solara

route_order = ["/","data","training","testing","inference"]

@solara.component
def Page():
    with solara.Row():
        solara.Markdown(md_text="""
## Welcome
This web page is created to demonstrate the model-based auto-scaling
approach developed in "AI-based Auto-Scaling and Tuning" project.
The model-based approach used a historical data containing the state
of the application such as replica and cpu-limit, as well as some metrics
measured from Prometheus.

* In [Data](/data) tab, you can investigate the raw data.
* In [Training](/training), you can build a model on the raw data.
* [Testing](/testing) tab is used to evaluate the performance of the trained model.
* Finally, [Inference](/inference) tab provides a simulation environment to test the model.

### Training
Training data is obtained from historical data collected from monitoring a target Kubernetes
deployment.
![Training](/static/public/training.png)
""")
