import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def corrs(df: pd.Series, x: str, y):
    if y != -1 and isinstance(y, str):
        return df[[x, y]].corr()
    else:
        return df.corr()[x].nlargest(5)


def comparison(df: pd.Series, cols: list):
    return df.describe()[cols][1:3]


def check_to_ignore(df: pd.DataFrame, level: str):
    cong = pd.crosstab(df['Walc'], df[level], margins=True)
    print(cong)

#
# def check_overlap(df: pd.DataFrame, x_covariates: []):
#     cm = CausalModel(Y=df['G3'].values, D=df['Walc'].values, X=x_covariates)
#     print(cm.summary_stats)


def histo(x: pd.Series, data_name):
    plt.xticks(ticks=list(range(min(x), max(x) + 1)))
    plt.hist(x, bins=(max(x)), range=(min(x), max(x) + 1))
    plt.ylabel('counts')
    plt.xlabel('column values')
    plt.title(f'`{x.name}` Distribution in {data_name} Dataset')
    plt.savefig(f'{x.name} Distribution in {data_name}.png')
    plt.show()
    plt.close()


def collapse_categorial(x, y):
    cong = pd.crosstab(x, y, margins=True)
    coll = []
    for i in cong.columns:
        for j in cong.index:
            if cong.at[j, i] < 5:
                coll.append((j, i))
    cong['3'] = cong['3'] + df['4'] + df['5']
    return cong


# chi-Test does not work, not enough sampels
def chi_ind_test(x, y):
    chiRes = stats.chi2_contingency((x, y))
    collapse_categorial(x, y)
    print(x.size)
    print(sum(x.shape))
    print(x.ndim)
    print(f'chi-square statistic: {chiRes[0]}')
    print(f'p-value: {chiRes[1]}')
    print(f'DOF: {chiRes[2]}')  # This is not componenes


if __name__ == '__main__':

    for file in ['student-mat.csv', 'student-por.csv']:
        data_name = 'Portuguese' if file == 'student-por.csv' else 'Math'
        df = pd.read_csv(file)
        for col in ['Walc', 'Dalc', 'goout', 'G3']:
            histo(df[col], data_name)
        print(f'{data_name} Correlations:')
        print(df[['G3', 'goout', 'Dalc', 'Walc']].corr())
        print(comparison(df, ['G3', 'G2', 'G1']))
        # check_to_ignore(df, 'Dalc')
        x_covariates = list(df.columns.difference(["G3", "Walc"]))
        # check_overlap(df, x_covariates)
        # # print(chi_ind_test(df['Walc'], df['Dalc']))