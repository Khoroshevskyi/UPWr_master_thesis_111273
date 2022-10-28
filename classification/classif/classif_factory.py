# Run together
from typing import NoReturn, Tuple
import pandas as pd
import logmuse
import coloredlogs

from sklearn.model_selection import train_test_split

from feature_selection import selection_dict
from normalization import normal_funct
from dim_red import feature_select_methods
from classif import classifiers_dict



_LOGGER = logmuse.init_logger(name="analysis")

coloredlogs.install(
    logger=_LOGGER,
    datefmt="%H:%M:%S",
    fmt="[%(levelname)s] [%(asctime)s] %(message)s",
)

# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=15)
# print(f'Train: {X_train.shape}')
# print(f'Test: {X_test.shape}')


def read_data(data_path: str = "/home/bnt4me/MasterTh/gdc_data/last_file.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read and sort the dataframe
    :param data_path: path to the data
    :return: x, y
    """
    data = pd.read_csv(data_path, delimiter="\t", index_col=0, low_memory=False)

    _LOGGER.info(f"Data numbers: {data.tumor_stage.value_counts()}")

    data1 = data[data.tumor_stage != "stage_4"]
    y = data1.tumor_stage
    x = data1.drop(['tumor_stage'], axis=1)

    _LOGGER.info(f"y unique check (before replace): {y.unique()}")

    y.replace({"stage_1": 0}, inplace=True)
    y.replace({"stage_2": 1}, inplace=True)
    y.replace({"stage_3": 2}, inplace=True)

    _LOGGER.info(f"X null check: {x.isnull().sum().sum()}")
    _LOGGER.info(f"y unique check: {y.unique()}")

    return x, y




def run_all_classification():
    list_of_results = []
    # 1
    _LOGGER.info(f"Reading data...")
    x, y = read_data()
    # 2
    _LOGGER.info(f"Splitting data for training and test data...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15)
    _LOGGER.info(f"Train data shape: {x_train.shape}")

    result_dict = {
        "feature_selection": "",
        "normalization": "",
        "dim_red": "",
        "classification": "",
        "status": "",

    }
    # 3 Feature selection:
    _LOGGER.info(f"Selecting the data with Feature selection ...")

    __selection = ""
    __norm = ""
    __dim = ""
    __status = ""

    for select in selection_dict:
        __selection = select
        try:
            _LOGGER.info(f"Feature selection function: {select}")
            processed_data = selection_dict[select](x_train, y_train)

            # 4 data normalization
            _LOGGER.info(f"Normalizing the data ...")

            for norm_f in normal_funct:
                __norm = norm_f
                try:
                    _LOGGER.info(f"Norm function: {norm_f}")
                    data_after_norm = normal_funct[norm_f](processed_data)

                    _LOGGER.info(f"Running Dim Reduction:")
                    try:
                        for dim_red_func in feature_select_methods:
                            __dim = dim_red_func
                            _LOGGER.info(f"Dim red function: {dim_red_func}")
                            data_after_dim = feature_select_methods[dim_red_func](data_after_norm)

                            _LOGGER.info(f"Now goes classification:")
                            for classifyer in classifiers_dict:
                                __class = classifyer
                                _LOGGER.info(f"Dim red function: {dim_red_func}")
                                classifiers_dict[classifyer](data_after_dim, y_train)



                            list_of_results.append(
                                {
                                    "feature_selection": __selection,
                                    "normalization": __norm,
                                    "dim_red": __dim,
                                    "classification": "",
                                    "f1-score": "",
                                    "status": "Pass",
                                })
                    except Exception:
                        list_of_results.append(
                            {
                                "feature_selection": __selection,
                                "normalization": __norm,
                                "dim_red": __dim,
                                "classification": "",
                                "f1-score": "",
                                "status": "Error",
                            })

                except Exception:
                    list_of_results.append(
                        {
                            "feature_selection": __selection,
                            "normalization": __norm,
                            "dim_red": __dim,
                            "classification": "",
                            "f1-score": "",
                            "status": "Error",
                        })

        except Exception:
            list_of_results.append(
                {
                    "feature_selection": __selection,
                    "normalization": __norm,
                    "dim_red": __dim,
                    "classification": "",
                    "f1-score": "",
                    "status": "Error",
                }
            )
    print(list_of_results)



    # 2 Dim reduction

    # 4 Classification
    # 5 Saving outputs


run_all_classification()

