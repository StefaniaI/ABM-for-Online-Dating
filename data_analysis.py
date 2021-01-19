import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(file_name):
    ''' Reads a csv file with statistics results & creates a data frame.'''
    df = pd.read_csv(file_name)
    df.drop_duplicates(keep=False, inplace=True)
    # df.norm_out_bias_value.unique()
    df = df[df['norm_out_bias_value'] == 0.8]
    df = df[df['beta'] == 0.2]
    df = df[df['gamma'] == 0.2]
    df = df[df['attribute_importance'] == 5]
    df = df[df['p_negativity'] == 1]
    df = df[df['no_matching_experiential'] == 1]
    df = df[df['no_competing_experiential'] == 1]
    df = df[df['no_competing_experiential'] == 1]
    df = df[df['strength_update_pref'] == 1]
    df = df[df['strength_update_norms'] == 0.1]

    colms = ['norm_out_bias_value', 'beta', 'gamma', 'gamma_m', 'gamma_c']
    colms += ['attribute_importance', 'p_negativity',
              'no_matching_experiential', 'no_competing_experiential', 'strength_update_pref', 'strength_update_norms']
    df = df.drop(columns=colms)

    return df


def plot_histogram(df):
    '''Plots the percentage of out-group long-term rel.s depending on the
    filter intervention and ethnocentrism degree for given data frame.
    Type - bar plot'''

    # df = pd.read_csv(file_name)
    # crit = abs(df['no_matching_searchable']-att[0][0]) + \
    #    abs(df['no_competing_searchable']-att[0][1]) == 0
    # crit = df['no_matching_searchable'] + df['no_competing_searchable'] >= 2
    # df = df[crit]
    polt_old = "percentage_outgroup_long_term"
    polt = 'Percentage of out-group\n long-term relationships'
    ni_old = "norm_intervention"
    ni = 'Norm intervention'
    fr_old = "filter"
    fr = 'Filter intervention'
    df = df.rename(columns={polt_old: polt, ni_old: ni, fr_old: fr})
    # df = df[df['no_competing_experiential'] == 2]
    df['Ethnocentrism degree'] = df['norm_out_bias_value']
    no_a = 'Ethnocentrism degree'
    df[ni] = df[ni].replace(0, 'Off')
    df[ni] = df[ni].replace(1, 'On')
    df[fr] = df[fr].replace('non_bias_WEAK', 'Weak non-bias')
    df[fr] = df[fr].replace('non_bias_STRONG', 'Strong non-bias')
    df[fr] = df[fr].replace('STRONG', 'Strong')
    df[fr] = df[fr].replace('OFF', 'Off')
    df[fr] = df[fr].replace('WEAK', 'Weak')
    ordered_f = ['Weak non-bias', 'Weak', 'Strong', 'Strong non-bias', 'Off']

    sns.set(style="ticks", rc={'axes.facecolor': '#F2F2F2',
                               'figure.facecolor': '#F2F2F2', 'patch.facecolor': '#F2F2F2'})
    fig = sns.catplot(x=fr, y=polt,
                      hue=ni, kind="bar", col=no_a, data=df, order=ordered_f,
                      legend_out=False, palette=sns.color_palette(['#437CDF', '#70AD47']))

    # change y axis to percentage
    from matplotlib.ticker import PercentFormatter
    for ax in fig.axes.flat:
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    def caption(fig):
        fig.set(xlabel='Filter intervention',
                ylabel='Percentage of out-group\n long-term relationships')

        # change legend
        # title
        new_title = 'Norm intervention'
        fig._legend.set_title(new_title)
        # replace labels
        new_labels = ['On', 'Off']
        for t, l in zip(fig._legend.texts, new_labels):
            t.set_text(l)

    # caption(fig)


def plot_number_long_term(df_orig, to_plot='no_long_term'):
    ''' Makes bar plots for different statistic depending on the
            - norm intervention, and
            - filter intervention.
    It plots one of the following (depending on to_plot):
         - # out-group long-term relationships
         - # long-term relationships
         - % relationships failing after first offline interaction
         - % time spent offline
         - % time spent searching
         - % of exits because of un-successful search turns
    '''

    # choose only standard values
    df = pick_standard_9_attributes(df_orig)
    if len(df) != 400:
        print("Taking the standard attributes doesn't give the right number of simulations!!!")

    lt_old = "no_long_term"
    lt = 'Number of \n long-term relationships'
    oglt_old = "no_outgroup_long_term"
    oglt = 'Number of outgroup \n long-term relationships'
    ni_old = "norm_intervention"
    ni = 'Norm intervention'
    fr_old = "filter"
    fr = 'Filter intervention'
    time_of_old = 'time_offline'
    time_of = 'Percentage of\n time spent offline'
    time_sch = 'time_searching'
    df = df.rename(columns={lt_old: lt, oglt_old: oglt,
                            ni_old: ni, fr_old: fr, time_of_old: time_of})
    # df['Ethnocentrism degree'] = df['norm_out_bias_value']
    no_a = 'Number of attributes'
    df[no_a] = 1+4*df['no_matching_searchable']
    df[ni] = df[ni].replace(0, 'Off')
    df[ni] = df[ni].replace(1, 'On')
    df[fr] = df[fr].replace('non_bias_WEAK', 'Weak\n non-race')
    df[fr] = df[fr].replace('non_bias_STRONG', 'Strong\n non-race')
    df[fr] = df[fr].replace('STRONG', 'Strong')
    df[fr] = df[fr].replace('OFF', 'Off')
    df[fr] = df[fr].replace('WEAK', 'Weak')
    ordered_f = ['Strong', 'Strong\n non-race', 'Weak', 'Weak\n non-race', 'Off']
    df[ni] = df[ni].replace(0, 'Off')
    df[ni] = df[ni].replace(1, 'On')
    fd = 'no_first_date'
    sd = 'no_second_date'
    disat = 'Percentage of successful 1st relationships'
    df[disat] = df[sd]/df[fd]
    df['per_high'] = df['no_oglt_high']/(df['no_oglt_low']+df['no_oglt_med']+df['no_oglt_high'])
    df['Percentage of exits from\n un-successful search turns'] = df['exit_reason_bad_rec'] / \
        (df['exit_reason_bad_rec']+df['exit_reason_failed_rel']+df['exit_reason_long_term'])

    # sns.set(style="ticks", rc={'axes.facecolor': '#F2F2F2',
    #                           'figure.facecolor': '#F2F2F2', 'patch.facecolor': '#F2F2F2'})

    sns.reset_orig()
    if to_plot == 'no_outgroup_long_term':
        fig = sns.catplot(x=fr, y=oglt,
                          hue=ni, kind="bar", data=df, order=ordered_f,
                          legend_out=False, palette=sns.color_palette(['#437CDF', '#70AD47']))
    elif to_plot == 'no_long_term':
        fig = sns.catplot(x=fr, y=lt,
                          hue=ni, kind="bar", data=df, order=ordered_f,
                          legend_out=False, palette=sns.color_palette(['#437CDF', '#70AD47']))
    elif to_plot == 'perc_failed_first_date':
        fig = sns.catplot(x=fr, y=disat,
                          hue=ni, kind="bar", data=df, order=ordered_f,
                          legend_out=False, palette=sns.color_palette(['#437CDF', '#70AD47']))
        # change y axis to percentage
        from matplotlib.ticker import PercentFormatter
        for ax in fig.axes.flat:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    elif to_plot == 'Percentage of\n time spent offline':
        sns.set_context("paper", font_scale=1)
        sns_plot = sns.catplot(x=fr, y=time_of, hue=ni, col=no_a,
                               palette={"On": '#70AD47', "Off": '#437CDF'},
                               markers=["^", "x"],
                               kind="point", order=ordered_f, data=df,
                               legend=False, dodge=True, label="Total", height=2.7, aspect=2.9/2.7)
        # change y axis to percentage
        from matplotlib.ticker import PercentFormatter
        for ax in sns_plot.axes.flat:
            ax.yaxis.set_major_formatter(PercentFormatter(1))

        # legend
        plt.legend(loc='lower right', title='Norm intervention')

        sns_plot.savefig("time_offline.png", dpi=300)
    elif to_plot == 'time_search':
        fig = sns.catplot(x=fr, y=time_sch,
                          hue=ni, kind="bar", data=df,  order=ordered_f,
                          legend_out=False, palette=sns.color_palette(['#437CDF', '#70AD47']))
        # change y axis to percentage
        from matplotlib.ticker import PercentFormatter
        for ax in fig.axes.flat:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    elif to_plot == 'per_high':
        fig = sns.catplot(x=fr, y='per_high',
                          hue=ni, kind="bar", data=df,  order=ordered_f,
                          legend_out=False, palette=sns.color_palette(['#437CDF', '#70AD47']))
        # change y axis to percentage
        from matplotlib.ticker import PercentFormatter
        for ax in fig.axes.flat:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    elif to_plot == 'exit_reason':
        sns.set_context("paper", font_scale=1)
        sns_plot = sns.catplot(x=fr, y='Percentage of exits from\n un-successful search turns', hue=ni, col=no_a,
                               palette={"On": '#70AD47', "Off": '#437CDF'},
                               markers=["^", "x"],  # linestyles=['None', 'None'],
                               kind="point", order=ordered_f, data=df,
                               legend=False, dodge=True, label="Total", height=2.7, aspect=2.9/2.7)
        # change y axis to percentage
        from matplotlib.ticker import PercentFormatter
        for ax in sns_plot.axes.flat:
            ax.yaxis.set_major_formatter(PercentFormatter(1))

        # sns_plot.lines[0].set_color('black')

        # legend
        plt.legend(loc='upper right', title='Norm intervention')

        sns_plot.savefig("exit_reason.png", dpi=300)

    plt.show()


def plot_all(df):
    ''' Plots all the options in the above 2 functions'''
    plot_histogram(df)
    plot_number_long_term(df, to_plot='no_long_term')
    plot_number_long_term(df, to_plot='no_outgroup_long_term')
    plot_number_long_term(df, to_plot='perc_failed_first_date')
    plot_number_long_term(df, to_plot='Percentage of\n time spent offline')
    plot_number_long_term(df, to_plot='time_search')
    plot_number_long_term(df, to_plot='exit_reason')
    plot_number_long_term(df, to_plot='per_high')

    plt.show()


def rename_cols(df):
    ''' Does the renaming for better labelling in plots'''
    lt_old = "no_long_term"
    lt = 'Number of \n long-term relationships'
    oglt_old = "no_outgroup_long_term"
    oglt = 'Number of outgroup \n long-term relationships'
    ni_old = "norm_intervention"
    ni = 'Norm intervention'
    fr_old = "filter"
    fr = 'Filter intervention'
    time_of = 'Percentage of\n time spent offline'
    time_sch = 'time_searching'
    polt_old = "percentage_outgroup_long_term"
    polt = 'Percentage of out-group\n long-term relationships'
    df = df.rename(columns={polt_old: polt, lt_old: lt, oglt_old: oglt, ni_old: ni, fr_old: fr})
    # df['Number of attributes'] = df['no_matching_searchable']
    no_a = 'Number of attributes'
    df[ni] = df[ni].replace(0, 'Off')
    df[ni] = df[ni].replace(1, 'On')
    df[fr] = df[fr].replace('non_bias_WEAK', 'Weak non-bias')
    df[fr] = df[fr].replace('non_bias_STRONG', 'Strong non-bias')
    df[fr] = df[fr].replace('STRONG', 'Strong')
    df[fr] = df[fr].replace('OFF', 'Off')
    df[fr] = df[fr].replace('WEAK', 'Weak')

    # df[polt] = 100*df[polt]

    return df


def pick_standard(df):
    '''Picks only the columns with the default parameter values'''
    df = df[df['no_matching_searchable'] == 1]
    df = df[df['no_matching_experiential'] == 1]
    df = df[df['no_competing_searchable'] == 1]
    df = df[df['no_competing_experiential'] == 1]

    df = df[df['beta'] == 0.4]
    df = df[df['gamma'] == 0.2]
    df = df[df['p_negativity'] == 0.25]
    df = df[df['strength_update_norms'] == 0.01]
    df = df[df['bad_recommandation_tolerance'] == 25]
    df = df[df['norm_out_bias_value'] == 0.2]
    df = df[df['strength_update_norms'] == 0.01]

    return df


def pick_standard_9_attributes(df):
    '''Picks only the columns with the default parameter values'''
    # 5 or 9 attributes
    df = df[(df['no_matching_searchable'] + df['no_matching_experiential'] +
             df['no_competing_searchable'] + df['no_competing_experiential']) % 4 == 0]

    df = df[df['beta'] == 0.4]
    df = df[df['gamma'] == 0.2]
    df = df[df['p_negativity'] == 0.25]
    df = df[df['strength_update_norms'] == 0.01]
    df = df[df['bad_recommandation_tolerance'] == 25]
    df = df[df['norm_out_bias_value'] == 0.2]
    df = df[df['strength_update_norms'] == 0.01]

    return df


def plot_pr_or_no(df_orig, to_plot='pr'):
    ''' Plots the percentage of long-term out-group for 5 and 9 attributes'''

    # pick the standard attributes
    if to_plot == 'pr':
        df = pick_standard_9_attributes(df_orig)
        if len(df) != 400:
            print("Taking the standard attributes doesn't give the right number of simulations!!!")
    elif to_plot == 'no':
        df = pick_standard(df_orig)
        if len(df) != 200:
            print("Taking the standard attributes doesn't give the right number of simulations!!!")

    lt = 'Number of \n long-term relationships'
    oglt = 'Number of outgroup \n long-term relationships'
    ni = 'Norm intervention'
    fr = 'Filter intervention'
    time_of = 'Percentage of\n time spent offline'
    time_sch = 'time_searching'
    polt = 'Percentage of out-group\n long-term relationships'
    df = rename_cols(df)
    df[fr] = df[fr].replace('Weak non-bias', 'Weak\n non-race')
    df[fr] = df[fr].replace('Strong non-bias', 'Strong\n non-race')
    ordered_f = ['Strong', 'Strong\n non-race', 'Weak', 'Weak\n non-race', 'Off']

    df['Number of attributes'] = 1 + 4*df['no_matching_searchable']
    no_a = 'Number of attributes'

    sns.set_style("white")
    # sns.set(rc={'figure.figsize': (4, 1)})

    if to_plot == 'pr':
        sns.set_context("paper", font_scale=1)
        sns_plot = sns.catplot(x=fr, y=polt, hue=ni, col=no_a,
                               palette={"On": '#70AD47', "Off": '#437CDF'},
                               markers=["^", "x"],
                               kind="point", order=ordered_f, data=df,
                               legend=False, dodge=True, label="Total", height=2.7, aspect=2.9/2.7)

        # change y axis to percentage
        from matplotlib.ticker import PercentFormatter
        for ax in sns_plot.axes.flat:
            ax.yaxis.set_major_formatter(PercentFormatter(1))

        # legend
        plt.legend(loc='lower right', title='Norm intervention')

        sns_plot.savefig("pr_outgroup_longterm.png", dpi=300)
    elif to_plot == 'no':
        sns.reset_orig()
        sns.set_context("paper", font_scale=1)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.7, 2.7))
        # 7.83 to 2.7 if same size
        g1 = sns.pointplot(x=fr, y=lt, hue=ni,
                           palette={"On": '#70AD47', "Off": '#437CDF'},
                           markers=["^", "x"],
                           kind="point", order=ordered_f, data=df,
                           legend_out=False, dodge=True, ax=ax[0], label="Total", height=1, aspect=1.4/1)

        g2 = sns.pointplot(x=fr, y=oglt, hue=ni,
                           palette={"On": '#70AD47', "Off": '#437CDF'},
                           markers=["^", "x"],
                           kind="point", order=ordered_f, data=df,
                           legend_out=False, dodge=True, ax=ax[1], label="Total", height=1, aspect=1.4/1)
        g2.get_legend().remove()
        g1.legend(loc='upper right', title='Norm intervention')
        plt.tight_layout()
        plt.show()
        fig.savefig("no_longterm.png", dpi=300)

    plt.show()


def plot_no_and_pr(df):
    '''Plots the number of long-term, out-group long-term, and percentage
    next to each other'''

    lt = 'Number of \n long-term relationships'
    oglt = 'Number of outgroup \n long-term relationships'
    ni = 'Norm intervention'
    fr = 'Filter intervention'
    time_of = 'Percentage of\n time spent offline'
    time_sch = 'time_searching'
    polt = 'Percentage of out-group\n long-term relationships'
    df = rename_cols(df)
    df[fr] = df[fr].replace('Weak non-bias', 'Weak\n non-race')
    df[fr] = df[fr].replace('Strong non-bias', 'Strong\n non-race')
    ordered_f = ['Strong', 'Strong\n non-race', 'Weak', 'Weak\n non-race', 'Off']

    fig, ax = plt.subplots(nrows=1, ncols=3)
    g1 = sns.pointplot(x=fr, y=lt, hue=ni,
                       palette={"On": '#70AD47', "Off": '#437CDF'},
                       markers=["^", "x"],
                       kind="point", order=ordered_f, data=df,
                       legend_out=False, dodge=True, ax=ax[0], label="Total")

    g2 = sns.pointplot(x=fr, y=oglt, hue=ni,
                       palette={"On": '#70AD47', "Off": '#437CDF'},
                       markers=["^", "x"],
                       kind="point", order=ordered_f, data=df,
                       legend_out=False, dodge=True, ax=ax[1], label="Total")

    g3 = sns.pointplot(x=fr, y=polt, hue=ni,
                       palette={"On": '#70AD47', "Off": '#437CDF'},
                       markers=["^", "x"],
                       kind="point", order=ordered_f, data=df,
                       legend_out=False, dodge=True, ax=ax[2], label="Total")
    g1.get_legend().remove()
    g2.get_legend().remove()
    g3.legend(loc='lower right', title='Norm intervention')
    plt.show()


def plot_number_long_term_comparative(df):
    ''' Plots both the number of out-group long term and the number of long term
    on the same graph'''

    from copy import deepcopy
    lt = 'Number of \n long-term relationships'
    oglt = 'Number of outgroup \n long-term relationships'
    ni = 'Norm intervention'
    fr = 'Filter intervention'
    time_of = 'Percentage of\n time spent offline'
    time_sch = 'time_searching'
    df = rename_cols(df)
    ordered_f = ['Weak non-bias', 'Weak', 'Strong', 'Strong non-bias', 'Off']

    fig, ax = plt.subplots()
    g1 = sns.pointplot(x=fr, y=lt, hue=ni,
                       palette={"On": '#70AD47', "Off": '#437CDF'},
                       markers=["^", "^"], linestyles=["-", "--"],
                       kind="point", order=ordered_f, data=df,
                       legend_out=False, dodge=True, ax=ax, label="Total")

    g2 = sns.pointplot(x=fr, y=oglt, hue=ni,
                       palette={"On": '#70AD47', "Off": '#437CDF'},
                       markers=["o", "o"], linestyles=["-", "--"],
                       kind="point", order=ordered_f, data=df,
                       legend_out=False, dodge=True, ax=ax, label="Out-group")

    # plt.gca().legend(ncol=2, title="Norm intervention:    Number of rel.: ")
    lines1 = [g2.lines[0]] + [g2.lines[6]]
    leg1 = ax.legend(lines1, ['Off', 'On'], title="Norm intervention", loc=(0.6, 0.475))
    lgnd = plt.legend()

    lgnd.legendHandles[0].set_color('black')
    lgnd.legendHandles[2].set_color('black')
    lines1 = [lgnd.legendHandles[0], lgnd.legendHandles[2]]
    leg2 = ax.legend(lines1, ['Total', 'Out-group'],
                     title="Number of relationships", loc=(0.6, 0.225))

    ax.add_artist(leg1)
    ax.add_artist(leg2)

    plt.show()
