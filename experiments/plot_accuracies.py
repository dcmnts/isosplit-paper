import bluster as bl
import pandas as pd
import altair as alt
from evaluate_clustering import evaluate_clustering


def plot_accuracies_vs_parameter(study: bl.Study, *, title: str, param_name: str='separation', param_title='Separation'):
    X = pd.DataFrame()
    X[param_name] = [ds.parameters[param_name] for ds in study.datasets]
    X['trial'] = [ds.parameters['trial'] for ds in study.datasets]
    algorithm_names = [c.name for c in study.datasets[0].clusterings]
    for a in algorithm_names:
        clusterings_for_alg = [[c for c in ds.clusterings if c.name == a][0]  for ds in study.datasets]
        accuracies_for_alg = [evaluate_clustering(ds.labels, clusterings_for_alg[i].labels) for i, ds in enumerate(study.datasets)]
        X[a] = accuracies_for_alg
    Y = X.melt(id_vars=[param_name, 'trial'], value_vars=algorithm_names, value_name='accuracy', var_name='algorithm')

    # algorithm selection
    selection = alt.selection_multi(fields=['algorithm'], toggle="true")

    # algorithm selector
    algorithm_df = pd.DataFrame({'algorithm': algorithm_names})
    # algorithm_condition = alt.condition(selection, alt.Color('algorithm:N'), alt.Opacity(0.2))
    algorithm_selector = alt.Chart(algorithm_df).mark_rect().encode(
        y='algorithm',
        color='algorithm:N',
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).add_selection(selection)

    # main chart
    opacity = alt.condition(selection, alt.value(1), alt.value(0.2))
    chart: alt.Chart = alt.Chart(Y).mark_line().encode(
        x=alt.X(param_name, title=param_title),
        y=alt.Y('mean(accuracy)', scale=alt.Scale(zero=False), title='Mean accuracy'),
        color=alt.Color('algorithm', legend=None),
        opacity=opacity
    ).properties(
        title=title,
        width=700,
        height=450
    )

    # In order to be able to control the marker size, I needed
    # to layer another chart
    chart = chart + alt.Chart(Y).mark_point(size=150, filled=True).encode(
        x=alt.X(param_name),
        y=alt.Y('mean(accuracy)'),
        color=alt.Color('algorithm', legend=None),
        tooltip=['algorithm', param_name, 'mean(accuracy)'],
        opacity=opacity
    )

    # composite chart
    chart = (algorithm_selector | chart)

    return chart
    
    # https://stackoverflow.com/questions/62601904/altair-saver-valueerror-unsupported-format-png
    # https://github.com/altair-viz/altair_saver/issues/104
    # chart.save(save_fname)