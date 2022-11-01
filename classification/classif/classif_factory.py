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
from sampling import sampling_func

from csv import DictWriter


_LOGGER = logmuse.init_logger(name="analysis")

coloredlogs.install(
    logger=_LOGGER,
    datefmt="%H:%M:%S",
    fmt="[%(levelname)s] [%(asctime)s] %(message)s",
)

# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=15)
# print(f'Train: {X_train.shape}')
# print(f'Test: {X_test.shape}')


class AddRecord:
    def __init__(self, file_path: str):
        self.file_path = file_path
        col_names = {"sampling": "sampling",
                     "feature_selection": "feature_selection",
                     "normalization": "normalization",
                     "dim_red": "dim_red",
                     "classification": "classification",
                     "precision_score": "precision_score",
                     "recall_score": "recall_score",
                     "f1_score": "f1_score",
                     "status": "status",
                     }
        self._fild_names = list(col_names.keys())

        self.append(col_names)


    def append(self,  record: dict):
        with open(self.file_path, 'a+') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=self._fild_names)

            _LOGGER.info(f"Writing new row: {record}")
            dictwriter_object.writerow(record)

            f_object.close()

add_record = AddRecord("try.csv")


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
    x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(x, y, random_state=15)
    _LOGGER.info(f"Train data shape: {x_train_n.shape}")

    _LOGGER.info(f"Selecting the data with Feature selection ...")

    __selection = ""
    __norm = ""
    __dim = ""
    __status = ""
    __sampling = ""

    for samling_i in sampling_func:
        __sampling = samling_i

        try:
            x_train, y_train = sampling_func[samling_i](x_train_n, y_train_n)

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
                            for dim_red_func in feature_select_methods:
                                __dim = dim_red_func
                                try:
                                    _LOGGER.info(f"Dim red function: {dim_red_func}")
                                    data_after_dim = feature_select_methods[dim_red_func](data_after_norm)

                                    _LOGGER.info(f"Now goes classification:")
                                    for classifyer in classifiers_dict:
                                        __class = classifyer
                                        try:
                                            _LOGGER.info(f"Dim red function: {dim_red_func}")
                                            check_score = classifiers_dict[classifyer](data_after_dim, y_train)

                                            add_record.append(
                                                {
                                                    "sampling": __sampling,
                                                    "feature_selection": __selection,
                                                    "normalization": __norm,
                                                    "dim_red": __dim,
                                                    "classification": __class,
                                                    "precision_score": check_score['mean_test_precision'],
                                                    "recall_score": check_score['mean_test_recall'],
                                                    "f1_score": check_score['f1'],
                                                    "status": "Pass",
                                                })
                                        except Exception:
                                            add_record.append({
                                                "sampling": __sampling,
                                                "feature_selection": __selection,
                                                "normalization": __norm,
                                                "dim_red": __dim,
                                                "classification": __class,
                                                "status": "Error in  classification",
                                            })

                                except Exception:
                                    add_record.append({
                                        "sampling": __sampling,
                                        "feature_selection": __selection,
                                        "normalization": __norm,
                                        "dim_red": __dim,
                                        "classification": "",
                                        "status": "Error in Dim Reduction",
                                    })

                        except Exception:
                            add_record.append(
                                {
                                    "sampling": __sampling,
                                    "feature_selection": __selection,
                                    "normalization": __norm,
                                    "dim_red": "",
                                    "classification": "",
                                    "status": "Error in Normalization",
                                })

                except Exception:
                    add_record.append(
                        {
                            "sampling": __sampling,
                            "feature_selection": __selection,
                            "normalization": "",
                            "dim_red": "",
                            "classification": "",
                            "status": "Error in selection",
                        }
                    )
        except Exception:
            add_record.append(
                {
                    "sampling": __sampling,
                    "feature_selection": __selection,
                    "normalization": "",
                    "dim_red": "",
                    "classification": "",
                    "status": "Error in sampling",
                }
)

run_all_classification()

