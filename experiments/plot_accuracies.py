import bluster as bl
import pandas as pd
import altair as alt
from evaluate_clustering import evaluate_clustering


def plot_accuracies_vs_separation(study: bl.Study, save_fname: str, *, title: str):
    X = pd.DataFrame()
    X['separation'] = [ds.parameters['separation'] for ds in study.datasets]
    X['trial'] = [ds.parameters['trial'] for ds in study.datasets]
    algorithm_names = [c.name for c in study.datasets[0].clusterings]
    for a in algorithm_names:
        clusterings_for_alg = [[c for c in ds.clusterings if c.name == a][0]  for ds in study.datasets]
        accuracies_for_alg = [evaluate_clustering(ds.labels, clusterings_for_alg[i].labels) for i, ds in enumerate(study.datasets)]
        X[a] = accuracies_for_alg
    Y = X.melt(id_vars=['separation', 'trial'], value_vars=algorithm_names, value_name='accuracy', var_name='algorithm')
    chart = alt.Chart(Y).mark_line(point=True).encode(
        x='separation',
        y='mean(accuracy)',
        color='algorithm'
    ).properties(
        title=title
    )
    # https://stackoverflow.com/questions/62601904/altair-saver-valueerror-unsupported-format-png
    # https://github.com/altair-viz/altair_saver/issues/104
    chart.save(save_fname)