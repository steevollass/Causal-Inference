import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as st

import warnings
warnings.filterwarnings('ignore')

def propensity_scores(df: pd.DataFrame, treatment_col: str, x_covariates: []):

    X = StandardScaler().fit_transform(df[x_covariates].values)
    treat = df[treatment_col].values

    model = LogisticRegression(max_iter=100000).fit(X, treat)

    return model.predict_proba(X)[:, 1]


def calculate_IPW(df: pd.DataFrame, treatment_col: str):
    cpy = df.copy()

    cpy['sum of y'] = df[treatment_col] * df['G3']

    numerator = ((1-cpy[treatment_col])*cpy['G3']*(1/(1-cpy['P']))).sum()
    denumerator = ((1-cpy[treatment_col])*(1/(1-cpy['P']))).sum()

    mean1 = cpy['sum of y'].sum()/cpy[treatment_col].sum()
    mean2 = numerator/denumerator
    return mean1 - mean2


def s_learner(df: pd.DataFrame, treatment_col: str, x_covariates: []):
    X = df[x_covariates + [treatment_col]].values
    X1 = df[df[treatment_col] == 1][x_covariates + [treatment_col]]
    X0 = df[df[treatment_col] == 0][x_covariates + [treatment_col]]
    y = df['G3']

    model = LinearRegression().fit(X, y)
    try:
        return (model.predict(X1.values).sum() - model.predict(X0.values).sum()) / len(X1)
    except ValueError:
        print('momoml')


def t_model(df: pd.DataFrame, treatment_col: str, x_covariates: [], t: int):
    X = df[df[treatment_col] == t][x_covariates].values
    y = df[df[treatment_col] == t]['G3'].values

    model_t = LinearRegression().fit(X, y)

    return model_t


def t_learner(df: pd.DataFrame, treatment_col: str, x_covariates: []):
    model1 = t_model(df, treatment_col, x_covariates, 1)
    model0 = t_model(df, treatment_col, x_covariates, 0)

    X1 = df[df[treatment_col] == 1][x_covariates].values

    return (model1.predict(X1).sum() - model0.predict(X1).sum()) / len(X1)


def matching_algo(df: pd.DataFrame, treatment_col: str, x_covariates: []):
    df1 = df[df[treatment_col] == 1][x_covariates + ['G3'] + ['P']]
    df0 = df[df[treatment_col] == 0][x_covariates + ['G3'] + ['P']]

    model = KNeighborsRegressor(n_neighbors=1)
    model.fit(df0[x_covariates].values, df0['G3'].values)

    return sum([y1 - model.predict([x1])[0] for (y1, x1) in zip(df1['G3'].values, df1[x_covariates].values)])\
           / len(df1[x_covariates].values)


def bootstrap(data: pd.DataFrame, treatment_col: str, x_covariates: [], B=150, sample_size=20):
    intervals = {'IPW': [], 'S-learner': [], 'T-learner': [], 'Matching': []}
    means = {'IPW': [], 'S-learner': [], 'T-learner': [], 'Matching': []}

    for i in range(1, 5):
        ATEs = {'IPW': [], 'S-learner': [], 'T-learner': [], 'Matching': []}

        for b in range(B):
            data_0 = data[data[treatment_col] == i].sample(sample_size, replace=True)
            data_0[treatment_col] -= i
            data_1 = data[data[treatment_col] == i + 1].sample(sample_size, replace=True)
            data_1[treatment_col] -= i

            input = pd.concat([data_0, data_1])

            ipw, s, t, match = calc_ATE(input, treatment_col, x_covariates)
            ATEs['IPW'].append(ipw)
            ATEs['S-learner'].append(s)
            ATEs['T-learner'].append(t)
            ATEs['Matching'].append(match)

        for key in ATEs:
            mean = np.mean(ATEs[key])
            means[key].append(mean)
            std = np.std(ATEs[key])
            size = st.t.ppf(0.975, B-1, mean, std) * std / np.sqrt(B - 1)
            interval = (mean - size, mean + size)
            intervals[key].append(interval)

    data_0 = data[data[treatment_col] == 1].sample(sample_size, replace=True)
    data_0[treatment_col] -= 1
    data_1 = data[data[treatment_col] == 5].sample(sample_size, replace=True)
    data_1[treatment_col] -= 4

    input = pd.concat([data_0, data_1])

    ipw, s, t, match = calc_ATE(input, treatment_col, x_covariates)
    final_ATEs = {'IPW': ipw, 'S-learner': s, 'T-learner': t, 'Matching': match}
    return means, intervals, final_ATEs


def calc_ATE(data: pd.DataFrame, treatment_col, x_covariates):

    data['P'] = propensity_scores(data, treatment_col, x_covariates)

    return calculate_IPW(data, treatment_col), s_learner(data, treatment_col, x_covariates), \
                                               t_learner(data, treatment_col, x_covariates),\
                                               matching_algo(data, treatment_col, x_covariates)


def plot_means(means, data_name, treatment_col):

    # for key in means:
    #     plt.plot(means[key])
    #     plt.xticks(ticks=[0, 1, 2, 3], labels=['1-2', '2-3', '3-4', '4-5'])
    #     plt.axhline(y=0, linestyle=':', c='black')
    #     plt.xlabel('Treatment Step')
    #     plt.ylabel('ATE')
    #     plt.title(f'ATE Calculated With {key}')
    #     plt.savefig(f'Plots/ATE_{key}.png')
    #     plt.close()

    for key in means:
        plt.plot(means[key], label=key)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['1-2', '2-3', '3-4', '4-5'])
    plt.axhline(y=0, linestyle=':', c='black')
    plt.xlabel('Treatment Step')
    plt.ylabel('ATE')
    plt.legend()
    plt.title(f'ATE Comparison on {data_name} dataset and T=`{treatment_col}`')
    plt.savefig(f'Plots/ATE_{data_name}_{treatment_col}.png')
    plt.close()


def plot_intervals(intervals, data_name, treatment_col):
    # for key in intervals:
    #     sizes = [x1 - x0 for x0, x1 in intervals[key]]
    #     plt.plot(sizes)
    #     plt.xticks(ticks=[0, 1, 2, 3], labels=['1-2', '2-3', '3-4', '4-5'])
    #     plt.xlabel('Treatment Step')
    #     plt.ylabel('Size')
    #     plt.title(f'Confidence Intervals Calculated With {key}')
    #     plt.savefig(f'Plots/Interval_{key}.png')
    #     plt.close()

    for key in intervals:
        sizes = [x1 - x0 for x0, x1 in intervals[key]]
        plt.plot(sizes, label=key)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['1-2', '2-3', '3-4', '4-5'])
    plt.xlabel('Treatment Step')
    plt.ylabel('Size')
    plt.legend()
    plt.title(f'Confidence Interval Size Comparison on {data_name} dataset and T=`{treatment_col}`')
    plt.savefig(f'Plots/Interval_{data_name}_{treatment_col}.png')
    plt.close()


def run_test(file: str, treatment_col: str):
    data_name = 'Portuguese' if file == 'student-por.csv' else 'Math'
    data = pd.read_csv(file)
    data = pd.get_dummies(data)
    x_covariates = list(data.columns.difference(['G3', 'Walc', 'Dalc']))

    means, intervals, final_ATEs = bootstrap(data, treatment_col, x_covariates)
    plot_means(means, data_name, treatment_col)
    plot_intervals(intervals, data_name, treatment_col)

    print(f'Dataset: {data_name}, Treatment:{treatment_col}')
    print(final_ATEs)


def main():
    for file in ['student-mat.csv', 'student-por.csv']:

        for treatment_col in ['Walc', 'Dalc']:
            run_test(file, treatment_col)


if __name__ == '__main__':
    main()