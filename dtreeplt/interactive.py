import ipywidgets as widgets
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

from dtreeplt import dtreeplt


def view_interactive(feature_names, target_names, X, y, clf):
    feature_buttons = []
    for feature in feature_names:
        feature_buttons.append(widgets.ToggleButton(
            value=True,
            description=feature,
            disabled=False,
            button_style='',
            tooltip='Description',
            icon='',
            layout=widgets.Layout(flex='1 1 auto', min_width='100px', width='auto')
            )
        )

    output = widgets.Output()

    def update_tree(change):

        with output:
            show_features = np.array(feature_names)[[button.value for button in feature_buttons]]

            # print(show_features)
            clf.fit(X[show_features], y)
            dtree = dtreeplt(model=clf, feature_names=show_features, target_names=target_names)
            clear_output()
            _ = dtree.view()
            plt.show()

    for button in feature_buttons:
        button.observe(update_tree, names='value', type='change')

    update_tree(None)

    box_layout = widgets.Layout(overflow_x='scroll',
                                flex_flow='wrap',
                                display='flex')
    return widgets.VBox([widgets.Label('Select Features: '), widgets.Box(feature_buttons, layout=box_layout),
                  widgets.Label('Decision Tree: '), output],
                 layout=widgets.Layout(min_height='500px')
                 )
