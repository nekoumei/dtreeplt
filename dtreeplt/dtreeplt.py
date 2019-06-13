import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class dtreeplt():
    '''
    Parameters
    ---------------
    model: sklearn.tree.DecisionTreeClassifier object
        You need prepare trained model.
        If it is None, use iris data and fitting automatically.
    feature_names: array like object(example numpy.array)
        list of feature names.
    target_names: array like object(example numpy.array)
        list of target names.
    filled: Bool
        If it is True, paint nodes to indicate majority class, like sklearn.
    X: numpy array or pandas DataFrame object
        It is necessary for interacitve mode.
    y: numpy array object
        It is necessary for interacitve mode.
    cmap: matplotlib cm object
        you can choose colormap for draw decision tree.
    eval: Bool
        if True, hold out 9:1 (stratified) and calc valid accuracy.
        the evaluation run only interactive mode.
    '''
    def __init__(self, model=None, X=None, y=None, feature_names=None, target_names=None,
                 filled=True, cmap=cm.Accent, eval=True):
        if model is None:
            print('Use Iris Datasets.')
            model = tree.DecisionTreeClassifier(min_samples_leaf=.1)
            X, y, feature_names, target_names = self._get_iris_data()
            model.fit(X, y)

        if type(X) == pd.core.frame.DataFrame:
            X = X.values
        elif type(X) == np.ndarray:
            pass
        elif type(X) == list:
            X = np.array(X)
        elif X is None:
            pass
        else:
            assert False, 'X must be pandas DataFrame, numpy array or list'

        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        self.filled = filled
        self.cmap = cmap
        self.eval = eval

    def _get_iris_data(self):
        from sklearn.datasets import load_iris

        data = load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names

        return X, y, feature_names, target_names

    def _get_tree_infomation(self):
        tree_info_dict = {}
        tree_info_dict['samples'] = self.model.tree_.n_node_samples
        tree_info_dict['values'] = self.model.tree_.value
        tree_info_dict['features'] = [self.feature_names[i] if i >=0 else 'target' for i in self.model.tree_.feature]
        tree_info_dict['thresholds'] = self.model.tree_.threshold
        tree_info_dict['impurities'] = self.model.tree_.impurity
        tree_info_dict['criterion'] = self.model.criterion
        tree_info_dict['node_count'] = self.model.tree_.node_count
        tree_info_dict['children_left'] = self.model.tree_.children_left
        tree_info_dict['children_right'] = self.model.tree_.children_right
        tree_info_dict['max_depth'] = self.model.tree_.max_depth

        return tree_info_dict

    def _get_class_names(self, values):
        class_ids = []
        for value in values:
            class_ids.append(np.argmax(value))
        classes = [self.target_names[i] for i in class_ids]
        return classes, class_ids

    def _calc_nodes_relation(self):
        tree_info_dict = self._get_tree_infomation()
        self.classes, self.class_ids = self._get_class_names(tree_info_dict['values'])
        links = []
        links_left = []
        links_right = []
        link = {}
        for i, child_left in enumerate(tree_info_dict['children_left']):
            if child_left != -1:
                link['source'] = i
                link['target'] = child_left
                links.append(link.copy())
                links_left.append(link.copy())
        for i, child_right in enumerate(tree_info_dict['children_right']):
            if child_right != -1:
                link['source'] = i
                link['target'] = child_right
                links.append(link.copy())
                links_right.append(link.copy())

        tree_info_dict['links'] = links

        tree_info_dict['nodes_height'] = self._calc_nodes_height(
            tree_info_dict['node_count'],
            tree_info_dict['max_depth'],
            tree_info_dict['links']
        )

        # 親ノード直下の子ノードがいくつあるか数える
        child_counts = []
        for i in range(tree_info_dict['node_count']):
            child_count = 0
            for link in tree_info_dict['links']:
                if link['source'] == i:
                    child_count += 1
            child_counts.append(child_count)

        # x軸の配置を調整する
        base_distance = 0.6
        append_coordinate = [base_distance * -1, base_distance]
        x_dict = {}
        for i in range(tree_info_dict['node_count']):
            x_dict[i] = None
        x_dict[0] = 0
        for i in range(tree_info_dict['node_count']):
            tmp = 0
            for link in tree_info_dict['links']:
                if link['source'] == i:
                    x = round(x_dict[link['source']] + append_coordinate[tmp], 5)
                    height = tree_info_dict['nodes_height'][link['target']]
                    for j, node_height in enumerate(tree_info_dict['nodes_height']):
                        try:
                            if (round(height, 1) == round(node_height, 1)) \
                                    and (round(x_dict[j], 1) == round(x, 1)):
                                x += base_distance * 2
                                x = round(x, 5)
                        except TypeError:
                            # None参照を無視する
                            pass

                    x_dict[link['target']] = round(x, 5)
                    tmp += 1

        # 親ノードと子ノードのx軸が離れすぎている場合調整する
        for link in tree_info_dict['links']:
            diff = x_dict[link['source']] - x_dict[link['target']]
            if round(diff, 1) < base_distance * -2:
                height = tree_info_dict['nodes_height'][link["target"]]
                for heigh in range(height + 1):
                    nodes = [i for i, x in enumerate(tree_info_dict['nodes_height']) if x == heigh]
                    for node in nodes:
                        x_dict[node] -= base_distance

        return x_dict, tree_info_dict

    def _calc_nodes_height(self, node_count, max_depth, links):
        heights_list = list(range(node_count))
        heights_list[0] = max_depth
        k = [0]
        for i in range(node_count):
            tmp = []
            for link in links:
                if link['source'] in k:
                    heights_list[link['target']] = heights_list[link['source']] - 1
                    tmp.append(link['target'])
            k = tmp.copy()

        return heights_list

    def _get_texts(self, tree_info_dict):
        texts = []
        for i in range(tree_info_dict['node_count']):
            if not tree_info_dict['features'][i] == 'target':
                text = f'{tree_info_dict["features"][i]} <= {tree_info_dict["thresholds"][i]:,.2f}\n'
            else:
                text = ''
            text += f'{tree_info_dict["criterion"]} = {tree_info_dict["impurities"][i]:.2f}\n\
samples = {tree_info_dict["samples"][i]}\n\
values = {tree_info_dict["values"][i]}\n\
class = {self.classes[i]}'
            texts.append(text)

        return texts

    def draw_figure(self, x_dict, tree_info_dict):
        fig = plt.figure(
            figsize=[
                (max(x_dict.values()) - min(x_dict.values())) * 6,
                tree_info_dict['nodes_height'][0] * 5
            ]
        )
        ax = fig.add_subplot(111)
        # 不要な枠線、軸の削除
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.tick_params(labelbottom=False, bottom=False)
        ax.tick_params(labelleft=False, left=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xlim(min(x_dict.values()), max(x_dict.values()))

        viz_x = {}
        viz_y = {}
        for i in range(tree_info_dict['node_count']):
            viz_x[i] = x_dict[i]
            viz_y[i] = tree_info_dict['nodes_height'][i]

        rect_width = 1
        rect_height = 0.7

        texts = self._get_texts(tree_info_dict)

        if self.filled:
            colors = [self.cmap(self.class_ids[i]) for i in range(len(self.class_ids))]
        else:
            colors = ['white' for i in range(len(self.class_ids))]
        for i, text in enumerate(texts):
            # nodeを表す四角形の描画
            rectangle = mpatch.Rectangle(
                (viz_x[i], viz_y[i]),
                rect_width,
                rect_height,
                color=colors[i],
                alpha=1 - tree_info_dict['impurities'][i],
                ec='#000000'
            )
            ax.add_artist(rectangle)

            # node内のtextの描画
            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0
            ax.annotate(text, (cx, cy), color='black',
                        fontsize=20, ha='center', va='center')

        # 矢印の描画
        for link in tree_info_dict['links']:
            x = x_dict[link['source']] + rect_width / 2
            y = tree_info_dict['nodes_height'][link['source']]
            dx = x_dict[link['target']] + rect_width / 2
            dy = tree_info_dict['nodes_height'][link['target']] + rect_height
            ax.annotate(s='', xy=(dx, dy), xytext=(x, y),
                        xycoords='data',
                        arrowprops=dict(arrowstyle='->')
                        )

        ax.set_xlim(min(x_dict.values()), max(x_dict.values()) + rect_width)
        _ = ax.set_ylim(min(tree_info_dict['nodes_height']), max(tree_info_dict['nodes_height']) + rect_height)

        return fig

    def view(self, interactive=False):
        '''
        Parameters
        ---------------
        interactive: Bool

        return
        --------------
        if interactive:
            fig: ipywidgets.VBox object
        else:
            fig: matplotlib.figure object
        '''
        x_dict, tree_info_dict = self._calc_nodes_relation()

        if interactive:
            from . import interactive as it
            return it.view_interactive(self.feature_names, self.target_names, self.X, self.y, self.model, self.eval)

        else:
            fig = self.draw_figure(x_dict, tree_info_dict)
            return fig
