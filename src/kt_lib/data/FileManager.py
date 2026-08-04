import os
import json
import platform
import numpy as np


from kt_lib.utils.data_io import read_csv, read_json, write_json


class FileManager:
    raw_data_root = os.path.join("data", "raw_datasets")
    preprocessed_data_root = os.path.join("data", "processed_datasets")
    dataset_raw_path = {
        "assist2009": os.path.join(raw_data_root, "assist2009", "skill_builder_data.csv"),
        "assist2009-full": os.path.join(raw_data_root, "assist2009-full", "assistments_2009_2010.csv"),
        "assist2012": os.path.join(raw_data_root, "assist2012", "2012-2013-data-with-predictions-4-final.csv"),
        "assist2015": os.path.join(raw_data_root, "assistments15", "2015_100_skill_builders_main_problems.csv"),
        "assist2017": os.path.join(raw_data_root, "assistments17", "anonymized_full_release_competition_dataset.csv"),
        "edi2020-task1": os.path.join(raw_data_root, "edi2020", "Task_1"),
        "edi2020-task34": os.path.join(raw_data_root, "edi2020", "Task_3_4"),
        "edi2022": os.path.join(raw_data_root, "edi2022"),
        "SLP-bio": os.path.join(raw_data_root, "SLP"),
        "SLP-chi": os.path.join(raw_data_root, "SLP"),
        "SLP-eng": os.path.join(raw_data_root, "SLP"),
        "SLP-mat": os.path.join(raw_data_root, "SLP"),
        "SLP-his": os.path.join(raw_data_root, "SLP"),
        "SLP-geo": os.path.join(raw_data_root, "SLP"),
        "SLP-phy": os.path.join(raw_data_root, "SLP"),
        "slepemapy-anatomy": os.path.join(raw_data_root, "slepemapy-anatomy", "answers.csv"),
        "statics2011": os.path.join(raw_data_root, "statics2011", "AllData_student_step_2011F.csv"),
        "ednet-kt1": os.path.join(raw_data_root, "ednet-kt1"),
        "xes3g5m": os.path.join(raw_data_root, "xes3g5m"),
        "DBE-KT22": os.path.join(raw_data_root, "DBE-KT22"),
        "algebra2005": os.path.join(raw_data_root, "kdd_cup2010"),
        "algebra2006": os.path.join(raw_data_root, "kdd_cup2010"),
        "algebra2008": os.path.join(raw_data_root, "kdd_cup2010"),
        "bridge2algebra2006": os.path.join(raw_data_root, "kdd_cup2010"),
        "bridge2algebra2008": os.path.join(raw_data_root, "kdd_cup2010"),
        "junyi2015": os.path.join(raw_data_root, "junyi2015"),
        "poj": os.path.join(raw_data_root, "poj", "poj_log.csv")
    }

    data_preprocessed_dir = {
        "assist2009": os.path.join(preprocessed_data_root, "ASSIST2009"),
        "assist2009-full": os.path.join(preprocessed_data_root, "assist2009-full"),
        "assist2012": os.path.join(preprocessed_data_root, "assist2012"),
        "assist2015": os.path.join(preprocessed_data_root, "assistments15"),
        "assist2017": os.path.join(preprocessed_data_root, "assistments17"),
        "edi2020-task1": os.path.join(preprocessed_data_root, "edi2020-task1"),
        "edi2020-task34": os.path.join(preprocessed_data_root, "edi2020-task34"),
        "edi2022": os.path.join(preprocessed_data_root, "edi2022"),
        "SLP-bio": os.path.join(preprocessed_data_root, "SLP-bio"),
        "SLP-chi": os.path.join(preprocessed_data_root, "SLP-chi"),
        "SLP-eng": os.path.join(preprocessed_data_root, "SLP-eng"),
        "SLP-mat": os.path.join(preprocessed_data_root, "SLP-mat"),
        "SLP-his": os.path.join(preprocessed_data_root, "SLP-his"),
        "SLP-geo": os.path.join(preprocessed_data_root, "SLP-geo"),
        "SLP-phy": os.path.join(preprocessed_data_root, "SLP-phy"),
        "statics2011": os.path.join(preprocessed_data_root, "statics2011"),
        "ednet-kt1": os.path.join(preprocessed_data_root, "ednet-kt1"),
        "junyi2015": os.path.join(preprocessed_data_root, "junyi2015"),
        "poj": os.path.join(preprocessed_data_root, "poj"),
        "slepemapy-anatomy": os.path.join(preprocessed_data_root, "slepemapy-anatomy"),
        "xes3g5m": os.path.join(preprocessed_data_root, "xes3g5m"),
        "algebra2005": os.path.join(preprocessed_data_root, "algebra2005"),
        "algebra2006": os.path.join(preprocessed_data_root, "algebra2006"),
        "algebra2008": os.path.join(preprocessed_data_root, "algebra2008"),
        "bridge2algebra2006": os.path.join(preprocessed_data_root, "bridge2algebra2006"),
        "bridge2algebra2008": os.path.join(preprocessed_data_root, "bridge2algebra2008"),
        "DBE-KT22": os.path.join(preprocessed_data_root, "DBE-KT22")
    }

    setting_dir = "dataset/settings"
    file_settings_name = "setting.json"

    def __init__(self, root_dir, init_dirs=False):
        self.root_dir = root_dir
        if init_dirs:
            self.create_dirs()
        self.builtin_datasets = sorted(FileManager.data_preprocessed_dir.keys())

    def create_dirs(self):
        assert os.path.exists(self.root_dir), f"{self.root_dir} not exist"
        all_dirs = {
            os.path.join(self.root_dir, "data"),
            os.path.join(self.root_dir, FileManager.raw_data_root),
            os.path.join(self.root_dir, FileManager.preprocessed_data_root),
            os.path.join(self.root_dir, "data", "demo"),
            os.path.join(self.root_dir, "dataset"),
            os.path.join(self.root_dir, "dataset", "settings"),
            os.path.join(self.root_dir, "dataset", "saved_models"),
        }
        for relative_path in FileManager.dataset_raw_path.values():
            all_dirs.add(os.path.dirname(os.path.join(self.root_dir, relative_path)))
        for relative_path in FileManager.data_preprocessed_dir.values():
            all_dirs.add(os.path.join(self.root_dir, relative_path))
        if platform.system() == "Windows":
            all_dirs = list(sorted(all_dirs, key=lambda dir_str: len(dir_str.split("\\"))))
        else:
            all_dirs = list(sorted(all_dirs, key=lambda dir_str: len(dir_str.split("/"))))
        for dir_ in all_dirs:
            os.makedirs(dir_, exist_ok=True)

    def get_root_dir(self):
        return self.root_dir

    def get_dataset_raw_path(self, dataset_name):
        return os.path.join(self.root_dir, FileManager.dataset_raw_path[dataset_name])

    # ==================================================================================================================
    def get_preprocessed_dir(self, dataset_name):
        if not FileManager.data_preprocessed_dir.get(dataset_name, False):
            return os.path.join(self.root_dir, FileManager.preprocessed_data_root, dataset_name)
        return os.path.join(self.root_dir, FileManager.data_preprocessed_dir[dataset_name])

    def save_q_table(self, Q_table, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        Q_table_path = os.path.join(preprocessed_dir, f"Q_table.npy")
        np.save(Q_table_path, Q_table)

    def save_data_statics_processed(self, statics, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        statics_path = os.path.join(preprocessed_dir, f"statics_preprocessed.json")
        write_json(statics, statics_path)

    def save_data_statics_raw(self, statics, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        statics_path = os.path.join(preprocessed_dir, f"statics_raw.json")
        write_json(statics, statics_path)

    def save_data_id_map(self, all_id_maps, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        for id_map_name, id_map in all_id_maps.items():
            id_map_path = os.path.join(preprocessed_dir, f"{id_map_name}.csv")
            id_map.to_csv(id_map_path, index=False)

    def get_q_table(self, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        try:
            Q_table = np.load(os.path.join(preprocessed_dir, "Q_table.npy"))
        except FileNotFoundError:
            Q_table = None
        return Q_table

    def get_data_statics_processed(self, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        statics_path = os.path.join(preprocessed_dir, f"statics_preprocessed.json")
        statics = read_json(statics_path)
        return statics

    def get_preprocessed_path(self, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        return os.path.join(preprocessed_dir, f"data.txt")

    def get_concept_id2name(self, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        concept_id2name_path = os.path.join(preprocessed_dir, "concept_id2name_map.csv")
        return read_csv(concept_id2name_path)

    def get_concept_id_map(self, dataset_name, data_type):
        assert data_type in ["multi_concept", "single_concept"]
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        concept_id_map_path = os.path.join(preprocessed_dir, f"concept_id_map_{data_type}.csv")
        return read_csv(concept_id_map_path)

    # ==================================================================================================================
    def add_new_setting(self, setting_name, setting_info):
        setting_dir = os.path.join(self.root_dir, FileManager.setting_dir, setting_name)
        if os.path.exists(setting_dir) and os.path.isdir(setting_dir):
            return
        setting_path = os.path.join(setting_dir, FileManager.file_settings_name)
        os.mkdir(setting_dir)
        with open(setting_path, "w") as f:
            json.dump(setting_info, f, indent=2)

    def delete_old_setting(self, setting_old_name):
        setting_dir = os.path.join(self.root_dir, FileManager.setting_dir, setting_old_name)
        assert os.path.exists(setting_dir) or os.path.isdir(setting_dir), f"{setting_old_name} dir does not exist"
        os.rmdir(setting_dir)

    def get_setting_dir(self, setting_name):
        result_dir = os.path.join(self.root_dir, FileManager.setting_dir)
        setting_names = os.listdir(result_dir)
        assert setting_name in setting_names, f"{setting_name} dir does not exist"
        return os.path.join(result_dir, setting_name)

    def get_setting_file_path(self, setting_name):
        setting_dir = self.get_setting_dir(setting_name)
        return os.path.join(setting_dir, FileManager.file_settings_name)
    # ==================================================================================================================
