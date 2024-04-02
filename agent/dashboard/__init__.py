import solara

@solara.component
def Page():
    with solara.VBox() as main:
        solara.Text("Home")

    return main
