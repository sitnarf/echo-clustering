from functools import partial
from typing import Dict, Iterable

from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from toolz import identity

from functional import pipe
from utils import load_input_data, load_global_config

plot_folder = './data/clustering/plots'


def get_datasets(
    base: DataFrame,
    impute: bool = True,
) -> Dict[str, DataFrame]:
    label = 'ACV2'
    data = pipe(
        base,
        format_columns,
        partial(
            select_features,
            features=(
                'RWT', 'EM', 'LVMI', 'IVSD', 'PP', 'GS', 'LA_Adi', 'SBP', 'AM', 'LVPWD', 'MVE_VEL',
                'LVIDD', 'LA_GS', 'RMVEA', 'PR', 'SM', 'LA_Asi', 'REEM', 'REAM', 'IVRT', 'LAEDVi',
                'LAESVi', 'MVA_VEL', 'AO_DIAM', 'EF_MOD', 'LA_A_4ch', 'MV_DECT', 'ESV_MODI',
                'LA_EF_4ch', 'SV_MODI'
            )
        ),
        impute_missing if impute else identity,
    )
    data = DataFrame(
        StandardScaler().fit_transform(data),
        columns=data.columns,
        index=data.index,
    )
    return dict(
        base=base,
        clustering=data,
        clustering_correlated_removed=select_features(
            data, (
                'RWT', 'EM', 'LVMI', 'GS', 'SBP', 'AM', 'LVPWD', 'MVE_VEL', 'LVIDD', 'LA_GS',
                'RMVEA', 'PR', 'SM', 'REEM', 'IVRT', 'LAESVi', 'MVA_VEL', 'AO_DIAM', 'EF_MOD',
                'MV_DECT', 'ESV_MODI', 'LA_EF_4ch', 'SV_MODI'
            )
        ),
        varsellcm=select_features(
            data, (
                'RWT', 'EM', 'LVMI', 'GS', 'SBP', 'AM', 'LVPWD', 'MVE_VEL', 'LVIDD', 'LA_GS',
                'RMVEA', 'SM', 'REEM', 'IVRT', 'LAESVi', 'MVA_VEL', 'AO_DIAM', 'EF_MOD', 'MV_DECT',
                'ESV_MODI', 'LA_EF_4ch', 'SV_MODI'
            )
        ),
        manual=select_features(
            data_all, (
                'SBP', 'PP', 'LVMI', 'PR', 'REEM', 'ESV_MODI', 'LAESVI', 'LA_GS', 'MVE_VEL',
                'MVA_VEL', 'RMVEA', 'AM', 'EM', 'GS', 'MV_DECT'
            )
        ),
        feature_selection_subsets={
            'manual': [
                'SBP', 'PP', 'LVMI', 'PR', 'REEM', 'ESV_MODI', 'LAESVI', 'LA_GS', 'MVE_VEL',
                'MVA_VEL', 'RMVEA', 'AM', 'EM', 'GS', 'MV_DECT'
            ],
            'normalized_cut_15': [
                'PR', 'IVRT', 'LVIDD', 'PP', 'AO_DIAM', 'IVSD', 'LVMI', 'ESV_MODI', 'SV_MODI',
                'RWT', 'AM', 'REAM', 'SBP', 'LVPWD', 'SM'
            ],
        },
        varsellcm_importance=[
            ['Variables', 'Discrim. Power', 'Discrim. Power (%)', 'Discrim. Power (% cum)'],
            ['REAM', 817.64, 9.31, 9.31],
            ['LAEDVI', 699.19, 7.96, 17.27],
            ['EM', 629.51, 7.17, 24.44],
            ['LA_ADI', 596.19, 6.79, 31.23],
            ['RMVEA', 582.88, 6.64, 37.87],
            ['REEM', 481.04, 5.48, 43.34],
            ['LAESVI', 414.47, 4.72, 48.06],
            ['LVMI', 345.93, 3.94, 52.00],
            ['LA_ASI', 342.89, 3.90, 55.91],
            ['MVA_VEL', 335.27, 3.82, 59.72],
            ['IVSD', 309.54, 3.52, 63.25],
            ['LA_GS', 309.44, 3.52, 66.77],
            ['SBP', 295.45, 3.36, 70.14],
            ['LA_A_4CH', 284.82, 3.24, 73.38],
            ['LA_EF_4CH', 273.40, 3.11, 76.49],
            ['PP', 262.97, 2.99, 79.49],
            ['MV_DECT', 255.25, 2.91, 82.39],
            ['AM', 253.88, 2.89, 85.28],
            ['LVPWD', 244.71, 2.79, 88.07],
            ['RWT', 233.04, 2.65, 90.72],
            ['AO_DIAM', 162.76, 1.85, 92.58],
            ['MVE_VEL', 156.93, 1.79, 94.36],
            ['IVRT', 142.54, 1.62, 95.99],
            ['SM', 136.16, 1.55, 97.54],
            ['ESV_MODI', 114.66, 1.31, 98.84],
            ['LVIDD', 39.61, 0.45, 99.29],
            ['SV_MODI', 28.07, 0.32, 99.61],
            ['PR', 15.19, 0.17, 99.79],
            ['GS', 13.58, 0.15, 99.94],
            ['EF_MOD', 5.15, 0.06, 100.00],
        ],
        varsellcm_importance_k_2=[
            ['Variables', 'Discrim. Power', 'Discrim. Power (%)', 'Discrim. Power (% cum)'],
            ['REAM', 643.22, 9.79, 9.79],
            ['EM', 557.63, 8.49, 18.28],
            ['RMVEA', 428.47, 6.52, 24.81],
            ['LAEDVI', 396.74, 6.04, 30.85],
            ['REEM', 385.05, 5.86, 36.71],
            ['LA_ADI', 337.76, 5.14, 41.85],
            ['LA_GS', 293.81, 4.47, 46.33],
            ['IVSD', 282.91, 4.31, 50.63],
            ['MVA_VEL', 265.06, 4.04, 54.67],
            ['SBP', 251.84, 3.83, 58.50],
            ['MV_DECT', 251.50, 3.83, 62.33],
            ['LVMI', 249.36, 3.80, 66.13],
            ['LVPWD', 238.89, 3.64, 69.77],
            ['RWT', 232.57, 3.54, 73.31],
            ['PP', 231.99, 3.53, 76.84],
            ['LAESVI', 209.58, 3.19, 80.03],
            ['LA_A_4CH', 201.95, 3.07, 83.11],
            ['LA_EF_4CH', 185.32, 2.82, 85.93],
            ['AO_DIAM', 183.30, 2.79, 88.72],
            ['LA_ASI', 161.47, 2.46, 91.18],
            ['AM', 149.00, 2.27, 93.45],
            ['IVRT', 129.09, 1.97, 95.41],
            ['MVE_VEL', 128.61, 1.96, 97.37],
            ['SM', 116.14, 1.77, 99.14],
            ['ESV_MODI', 21.25, 0.32, 99.46],
            ['GS', 15.08, 0.23, 99.69],
            ['EF_MOD', 12.04, 0.18, 99.87],
            ['LVIDD', 8.23, 0.13, 100.00],
        ],
        label=label,
        y_true=base[label],
    )


def format_columns(data: DataFrame) -> DataFrame:
    data_new = data.copy()
    data_new.columns = [column.upper() for column in data.columns]
    return data_new


def select_features(data: DataFrame, features: Iterable[str]) -> DataFrame:
    return data[[feature.upper() for feature in features]]


def impute_missing(data: DataFrame) -> DataFrame:
    return DataFrame(SimpleImputer().fit_transform(data), columns=data.columns, index=data.index)


config = load_global_config()
data_all = load_input_data()
data_without_cardiac_events = data_all[data_all['HCAR2'] != 1]

datasets_all = get_datasets(data_all)
datasets_without_cardiac_events = get_datasets(data_without_cardiac_events)

datasets_to_report = [
    (
        'All—correlated >0.8 removed', 'all_without_correlation',
        datasets_all['clustering_correlated_removed']
    ),
]
