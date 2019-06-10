import ipywidgets as widgets
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dtreeplt import dtreeplt


def view_interactive(feature_names, target_names, X, y, clf, eval):
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

    if eval:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
    else:
        X_train, y_train = X, y

    def update_tree(change):
        with output:
            is_shows = [button.value for button in feature_buttons]
            show_features = np.array(feature_names)[is_shows]

            clf.fit(X_train[:, is_shows], y_train)

            if eval:
                y_pred = clf.predict(X_valid[:, is_shows])
                accuracy = accuracy_score(y_valid, y_pred)
            dtree = dtreeplt(model=clf, feature_names=show_features, target_names=target_names, X=X_train, y=y_train)
            clear_output()
            fig = dtree.view()
            if eval:
                fig.suptitle(f'Accuracy(Hold Out 9:1): {accuracy * 100:.3f}%', x=0, fontsize=20)
            plt.tight_layout()
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
