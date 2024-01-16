from scripts.templatemanager import StyleFile, FileManager, display_columns, columns, Additionals
import pandas as pd
import datetime

# t = tm('a.csv')
# data = t.data.copy()
# print(data)
# t.data = data
# t.save()
# data = FileManager.get_current_styles()
# print(data)

# FileManager.clear_style_cache()
# FileManager.update_additional_style_files()
# FileManager.get_current_styles()

# StyleFile.loaded_styles()

def handle_autosort_checkbox_change(data: pd.DataFrame, filename) -> pd.DataFrame:
    fm = FileManager()
    fm.create_file_if_missing(filename)

    sl = StyleFile(filename)
    old_data = fm.get_styles(filename)
    if old_data is not None:
        styles = old_data.drop(index=[i for (i, row) in old_data.iterrows() if Additionals.has_prefix(row[1])])
        styles = pd.concat([styles, data], ignore_index=True)
        styles['sort'] = [i + 1 for i in range(len(styles['sort']))]
    else:
        styles = data
    styles = sl.sort_dataset(styles)
    fm.save_current_styles(styles, filename)
    return data

# data = pd.DataFrame(data=[
#     ['模板3', 'http://1', '1,2,3,4,5', '分类1,', '参数1.', '评分：1', '备注111', datetime.datetime.now(
#                 tz=datetime.timezone(datetime.timedelta(hours=8))
#             ).strftime("%Y-%m-%d %H:%M:%S")],
# ], columns=columns)
# handle_autosort_checkbox_change(data, '123')


# from utils.image import image_to_base64
# print(image_to_base64('/tmp/gradio/22a64fb4a7b055c14704aabef9ae89dccdd54f0c/6.png', False))


try:
    raise Exception('123')
except Exception as e:
    raise e
finally:
    print(1111)