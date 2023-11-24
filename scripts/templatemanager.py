
import pandas as pd
from utils.constant import project_dir
# from modules.shared import cmd_opts, opts, prompt_styles
from pathlib import Path
import numpy as np
import os, json
from typing import Dict

import shutil
import datetime

try:
  import pyAesCrypt
except:
    pass

name_column = 'name'
# 录入
columns = [name_column,'模板图片', '模板尺寸', '模板形状', '模板坐标', '商品分类', '推理参数', '评分', '备注', 'date']
user_columns = [name_column,'模板图片', '模板尺寸', '模板形状', '模板坐标', '商品分类', '推理参数', '评分', '备注', 'date']
display_columns = ['sort', name_column,'模板图片', '模板尺寸', '模板形状', '模板坐标', '商品分类', '推理参数', '评分', '备注', 'date']
d_types = {name_column:str,'模板图片':str, '模板尺寸':str, '模板形状':str, '模板坐标':str, '商品分类':str, '推理参数':str, '评分':str, '备注':str, 'date':str}

base_dir = os.path.join(project_dir, "worker_data/template/styles/")


class StyleFile:
    def __init__(self, prefix: str):
        # Additionals.init(default_style_file_path=os.path.join(base_dir, "default.csv"),
        #                  additional_style_files_directory=base_dir)
        self.prefix = prefix
        # self.filename = f'{project_dir}/worker_data/template/'
        self.filename = Additionals.full_path(prefix)
        self.data: pd.DataFrame = self._load()

    def _load(self):
        try:
            data = pd.read_csv(self.filename, header=None, names=columns,
                               encoding="utf-8-sig", dtype=d_types,
                               skiprows=[0], usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        except:
            data = pd.DataFrame(columns=columns)

        indices = range(data.shape[0])
        data.insert(loc=0, column="sort", value=[i + 1 for i in indices])
        # data.update(loc=-1, column="date", value=datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"))
        data.fillna('', inplace=True)
        # data.insert(loc=4, column="notes",
        #             value=[FileManager.lookup_notes(data['name'][i], self.prefix) for i in indices])
        if len(data) > 0:
            for column in user_columns:
                data[column] = data[column].str.replace('\n', '<br>', regex=False)
        return data

    @staticmethod
    def sort_dataset(data: pd.DataFrame) -> pd.DataFrame:
        def _to_numeric(series: pd.Series):
            nums = pd.to_numeric(series)
            if any(nums.isna()):
                raise Exception("don't update display")
            return nums

        try:
            return data.sort_values(by='sort', axis='index', inplace=False, na_position='first', key=_to_numeric)
        except:
            return data

    def save(self):
        self.fix_duplicates()
        clone = self.data.copy()
        if len(clone) > 0:
            for column in user_columns:
                clone[column] = clone[column].str.replace('<br>', '\n', regex=False)
        clone.to_csv(self.filename, encoding="utf-8-sig", columns=columns, index=False)

    def fix_duplicates(self):
        names = self.data['name']
        used = set()
        for index, value in names.items():
            if value in used:
                while value in used:
                    value = value + "x"
                names.at[index] = value
            used.add(value)


import os


class Additionals:
    @classmethod
    def init(cls, default_style_file_path, additional_style_files_directory) -> None:
        cls.default_style_file_path = default_style_file_path
        cls.additional_style_files_directory = additional_style_files_directory

    @staticmethod
    def has_prefix(fullname: str):
        """
        Return true if the fullname is prefixed.
        """
        return ('::' in fullname)

    @staticmethod
    def split_stylename(fullname: str):
        """
        Split a stylename in the form [prefix::]name into prefix, name or None, name
        """
        if '::' in fullname:
            return fullname[:fullname.find('::')], fullname[fullname.find('::') + 2:]
        else:
            return None, fullname

    @staticmethod
    def merge_name(prefix: str, name: str):
        """
        Merge prefix and name into prefix::name (or name, if prefix is none or '')
        """
        if prefix:
            return prefix + "::" + name
        else:
            return name

    @staticmethod
    def prefixed_style(maybe_prefixed_style: str, current_prefix: str, force=False):
        """
        If force is False:
          If it has a prefix, return it.
          If not, use the one specified. If that is None or '', no prefix
        If force is True:
          use the prefix specified
        """
        prefix, style = Additionals.split_stylename(maybe_prefixed_style)
        prefix = current_prefix if force else (prefix or current_prefix)
        return Additionals.merge_name(prefix, style)

    @classmethod
    def full_path(cls, filename: str) -> str:
        """
        Return the full path for an additional style file.
        Input can be the full path, the filename with extension, or the filename without extension.
        If input is None, '', or the default style file path, return the default style file path
        """
        if filename is None or filename == '' or filename == cls.default_style_file_path:
            return cls.default_style_file_path
        filename = filename + ".csv" if not filename.endswith(".csv") else filename
        return os.path.relpath(os.path.join(cls.additional_style_files_directory, os.path.split(filename)[1]))

    @classmethod
    def display_name(cls, filename: str) -> str:
        """
        Return the full path for an additional style file.
        Input can be the full path, the filename with extension, or the filename without extension
        """
        fullpath = cls.full_path(filename)
        return os.path.splitext(os.path.split(fullpath)[1])[0] if fullpath != cls.default_style_file_path else ''

    @classmethod
    def additional_style_files(cls, include_new, display_names):
        format = cls.display_name if display_names else cls.full_path
        additional_style_files = [format(f) for f in os.listdir(cls.additional_style_files_directory) if
                                  f.endswith(".csv")]
        return additional_style_files + ["--Create New--"] if include_new else additional_style_files

    @classmethod
    def prefixes(cls):
        return cls.additional_style_files(include_new=False, display_names=True)


class FileManager:
    basedir = base_dir
    additional_style_files_directory = os.path.join(basedir, "additonal_style_files")
    backup_directory = os.path.join(basedir, "backups")
    if not os.path.exists(backup_directory):
        os.mkdir(backup_directory)
    if not os.path.exists(additional_style_files_directory):
        os.mkdir(additional_style_files_directory)

    default_style_file_path = basedir
    current_styles_file_path = default_style_file_path

    Additionals.init(default_style_file_path=default_style_file_path,
                     additional_style_files_directory=additional_style_files_directory)

    try:
        with open(os.path.join(basedir, "notes.json")) as f:
            notes_dictionary = json.load(f)
    except:
        notes_dictionary = {}

    encrypt = False
    encrypt_key = ""
    loaded_styles: Dict[str, StyleFile] = {}

    @classmethod
    def clear_style_cache(cls):
        """
        Drop all loaded styles
        """
        cls.loaded_styles = {}

    @classmethod
    def get_current_styles(cls, prefix=''):
        # return cls.get_styles(cls._current_prefix())
        return cls.get_styles(prefix)

    @classmethod
    def using_additional(cls):
        return cls._current_prefix() != ''

    @classmethod
    def get_styles(cls, prefix='') -> pd.DataFrame:
        """
        If prefix is '', this is the default style file.
        Load or retrieve from cache
        """
        if not prefix in cls.loaded_styles:
            cls.loaded_styles[prefix] = StyleFile(prefix)
        return cls.loaded_styles[prefix].data.copy()

    @classmethod
    def save_current_styles(cls, data, prefix=''):
        cls.save_styles(data, prefix,)
        # cls.save_styles(data, cls._current_prefix(),)

    @classmethod
    def save_styles(cls, data: pd.DataFrame, prefix=''):
        if not prefix in cls.loaded_styles:
            cls.loaded_styles[prefix] = StyleFile(prefix)
        cls.loaded_styles[prefix].data = data
        cls.loaded_styles[prefix].save()

        cls.update_notes_dictionary(data, prefix)
        cls.save_notes_dictionary()

        # prompt_styles.reload()

    @staticmethod
    def create_file_if_missing(filename):
        filename = Additionals.full_path(filename)
        if not os.path.exists(filename):
            print("", file=open(filename, "w"))

    @staticmethod
    def add_or_replace(array: np.ndarray, row):
        for i in range(len(array)):
            if array[i][1] == row[1]:
                array[i] = row
                return array
        return np.vstack([array, row])

    @classmethod
    def update_additional_style_files(cls):
        additional_files_as_numpy = {prefix: FileManager.get_styles(prefix=prefix).to_numpy() for prefix in
                                     Additionals.additional_style_files(include_new=False, display_names=True)}
        for _, row in cls.get_styles().iterrows():
            prefix, row[1] = Additionals.split_stylename(row[1])
            if prefix:
                if prefix in additional_files_as_numpy:
                    additional_files_as_numpy[prefix] = cls.add_or_replace(additional_files_as_numpy[prefix], row)
                else:
                    additional_files_as_numpy[prefix] = np.vstack([row])
        for prefix in additional_files_as_numpy:
            cls.save_styles(pd.DataFrame(additional_files_as_numpy[prefix], columns=display_columns), prefix=prefix)

    @classmethod
    def merge_additional_style_files(cls):
        styles = cls.get_styles('')
        styles = styles.drop(index=[i for (i, row) in styles.iterrows() if Additionals.has_prefix(row[1])])
        for prefix in Additionals.prefixes():
            styles_with_prefix = cls.get_styles(prefix=prefix).copy()
            if len(styles_with_prefix) == 0:
                os.remove(Additionals.full_path(prefix))
            else:
                styles_with_prefix[name_column] = [Additionals.merge_name(prefix, x) for x in
                                                   styles_with_prefix[name_column]]
                styles = pd.concat([styles, styles_with_prefix], ignore_index=True)
        styles['sort'] = [i + 1 for i in range(len(styles['sort']))]
        cls.save_styles(styles)

    @classmethod
    def _current_prefix(cls):
        return Additionals.display_name(cls.current_styles_file_path)

    @classmethod
    def move_to_additional(cls, maybe_prefixed_style, new_prefix):
        old_prefixed_style = Additionals.prefixed_style(maybe_prefixed_style, cls._current_prefix())
        new_prefixed_style = Additionals.prefixed_style(maybe_prefixed_style, new_prefix, force=True)
        data = cls.get_styles()
        data[name_column] = data[name_column].str.replace(old_prefixed_style, new_prefixed_style)
        cls.save_styles(data)
        cls.remove_from_additional(old_prefixed_style)
        cls.update_additional_style_files()

    @classmethod
    def remove_style(cls, maybe_prefixed_style):
        prefixed_style = Additionals.prefixed_style(maybe_prefixed_style, cls._current_prefix())
        data = cls.get_styles()
        rows_to_drop = [i for (i, row) in data.iterrows() if row[1] == prefixed_style]
        cls.save_styles(data.drop(index=rows_to_drop))
        cls.remove_from_additional(prefixed_style)
        cls.update_additional_style_files()

    @classmethod
    def duplicate_style(cls, maybe_prefixed_style):
        prefixed_style = Additionals.prefixed_style(maybe_prefixed_style, cls._current_prefix())
        data = cls.get_styles()
        new_rows = pd.DataFrame([row for (i, row) in data.iterrows() if row[1] == prefixed_style])
        data = pd.concat([data, new_rows], ignore_index=True)
        data = StyleFile.sort_dataset(data)
        cls.save_styles(data)
        cls.update_additional_style_files()

    @classmethod
    def remove_from_additional(cls, maybe_prefixed_style):
        prefix, style = Additionals.split_stylename(maybe_prefixed_style)
        if prefix:
            data = cls.get_styles(prefix)
            data = data.drop(index=[i for (i, row) in data.iterrows() if row[1] == style])
            cls.save_styles(data, prefix=prefix)

    @classmethod
    def do_backup(cls):
        fileroot = os.path.join(cls.backup_directory, datetime.datetime.now().strftime("%y%m%d_%H%M"))
        if not os.path.exists(cls.default_style_file_path):
            return
        shutil.copyfile(cls.default_style_file_path, fileroot + ".csv")
        paths = sorted(Path(cls.backup_directory).iterdir(), key=os.path.getmtime, reverse=True)
        for path in paths[24:]:
            os.remove(str(path))
        if cls.encrypt and len(cls.encrypt_key) > 0:
            try:
                pyAesCrypt.encryptFile(fileroot + ".csv", fileroot + ".csv.aes", cls.encrypt_key)
                os.remove(fileroot + ".csv")
            except:
                print("Failed to encrypt")

    @classmethod
    def list_backups(cls):
        return [file for file in os.listdir(cls.backup_directory) if (file.endswith('csv') or file.endswith('aes'))]

    @classmethod
    def backup_file_path(cls, file):
        return os.path.join(cls.backup_directory, file)

    @classmethod
    def restore_from_backup(cls, file):
        path = cls.backup_file_path(file)
        if not os.path.exists(path):
            return "Invalid selection"
        if os.path.splitext(file)[1] == ".aes":
            try:
                temp = os.path.join(cls.backup_directory, "temp.aes")
                temd = os.path.join(cls.backup_directory, "temp.csv")
                shutil.copyfile(file, temp)
                pyAesCrypt.decryptFile(temp, temd, cls.encrypt_key)
                os.rename(temd, cls.default_style_file_path)
            except:
                error = "Failed to decrypt .aes file"
            finally:
                if os.path.exists(temp):
                    os.remove(temp)
                if os.path.exists(temd):
                    os.remove(temd)
        else:
            shutil.copyfile(path, cls.default_style_file_path)
        return None

    @classmethod
    def restore_from_upload(cls, tempfile):
        error = None
        if os.path.exists(cls.default_style_file_path):
            if os.path.exists(cls.default_style_file_path + ".temp"):
                os.remove(cls.default_style_file_path + ".temp")
            os.rename(cls.default_style_file_path, cls.default_style_file_path + ".temp")
        if os.path.splitext(tempfile)[1] == ".aes":
            try:
                pyAesCrypt.decryptFile(tempfile, cls.default_style_file_path, cls.encrypt_key)
            except:
                error = "Failed to decrypt .aes file"
        elif os.path.splitext(tempfile)[1] == ".csv":
            os.rename(tempfile, cls.default_style_file_path)
        else:
            error = "Can only restore from .csv or .aes file"
        if os.path.exists(cls.default_style_file_path + ".temp"):
            if os.path.exists(cls.default_style_file_path):
                os.remove(cls.default_style_file_path + ".temp")
            else:
                os.rename(cls.default_style_file_path + ".temp", cls.default_style_file_path)
        return error

    @classmethod
    def save_notes_dictionary(cls):
        print(json.dumps(cls.notes_dictionary), file=open(os.path.join(cls.basedir, "notes.json"), 'w'))

    @classmethod
    def update_notes_dictionary(cls, data: pd.DataFrame, prefix: str):
        for _, row in data.iterrows():
            stylename = prefix + "::" + row[1] if prefix != '' else row[1]
            cls.notes_dictionary[stylename] = row[4]

    @classmethod
    def lookup_notes(cls, stylename, prefix):
        stylename = prefix + "::" + stylename if prefix != '' else stylename
        return cls.notes_dictionary[stylename] if stylename in cls.notes_dictionary else ''

import time, threading

class Background:
  """
  A simple background task manager that considers doing a background task every n seconds.
  The task is only done if the manager has been marked as pending.
  """
  def __init__(self, method, sleeptime) -> None:
    """
    Create a manager that will consider calling `method` every `sleeptime` seconds
    """
    self.method = method
    self.sleeptime = sleeptime
    self._pending = False
    self._started = False
    self.lock = threading.Lock()

  def start(self):
    """
    Start the manager's thread
    """
    with self.lock:
      if not self._started:
        threading.Thread(group=None, target=self._action, daemon=True).start()
        self._started = True

  def set_pending(self, pending=True):
    """
    Set the task as pending. Next time the manager checks it will call `method` and then unset pending.
    """
    with self.lock:
        self._pending = pending

  def _action(self):
    while True:
      with self.lock:
        if self._pending:
            self.method()
            self._pending = False
      time.sleep(self.sleeptime)




file_name='1123'


def add(data: pd.DataFrame, filename=file_name) -> pd.DataFrame:
    fm = FileManager()
    fm.create_file_if_missing(filename)

    sl = StyleFile(filename)
    old_data = fm.get_styles(filename)
    print("old_data:", old_data)
    if old_data is not None:
        styles = old_data.drop(index=[i for (i, row) in old_data.iterrows() if Additionals.has_prefix(row[1])])
        styles = pd.concat([styles, data], ignore_index=True)
        styles['sort'] = [i + 1 for i in range(len(styles['sort']))]
    else:
        print("old is None")
        styles = data
    styles = sl.sort_dataset(styles)
    fm.save_current_styles(styles, filename)
    return styles


def get_styles(filename=file_name):
    fm = FileManager()
    return fm.get_current_styles(filename)


def update_styles(data:pd.DataFrame, filename=file_name) -> pd.DataFrame:
    backup = Background(FileManager.do_backup, 600)
    backup.set_pending()
    print("data:", data)
    # data = StyleFile.sort_dataset(data) if autosort else data
    FileManager.save_current_styles(data, filename)
    return data

# data = pd.DataFrame(data=[
#     ['模板3', 'http://1', '1,2,3,4,5', '分类1,', '参数1.', '评分：1', '备注111', datetime.datetime.now(
#                 tz=datetime.timezone(datetime.timedelta(hours=8))
#             ).strftime("%Y-%m-%d %H:%M:%S")],
# ], columns=columns)
# add(data, 123)

