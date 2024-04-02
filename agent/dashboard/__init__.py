import solara

route_order = ["data","training","testing"]

@solara.component
def Page():
    with solara.VBox() as main:
        solara.Text("Home")

    return main
