
import uuid
import pathlib
import os
from utils.utils import project_dir


def generate_uuid():
    str = uuid.uuid4().hex
    return str[:8]


uuid = generate_uuid()

# 普通商品图
history = f"{project_dir}/worker_data/history"

# 服装商品图
dress_history = f"{project_dir}/worker_data/history/dress"
pathlib.Path(history).mkdir(parents=True, exist_ok=True)
pathlib.Path(dress_history).mkdir(parents=True, exist_ok=True)

# 默认场景图:需要初始化
deft_scene_image = f'{history}/def_scene_image/'

# 场景图>商品图>合并图>mask图（扣除背景制作mask）> mask图（mask反转）>生成图
# 商品图
commodity_image_dir = history + '/{uuid}/commodity_image'.format(uuid=uuid)
# 场景图
scene_image_dir = history + '/{uuid}/scene_image'.format(uuid=uuid)
# 商品图去背
commodity_rembg_image_dir = history + '/{uuid}/commodity_rembg_image'.format(uuid=uuid)
commodity_rembg_mask_image_dir = history + '/{uuid}/commodity_rembg_mask_image'.format(uuid=uuid)
# 合并图
commodity_merge_scene_image_dir = history + '/{uuid}/commodity_merge_scene_image'.format(uuid=uuid)
# 合并后的商品图 mask
merge_after_mask_image_dir = history + '/{uuid}/merge_after_mask_image'.format(uuid=uuid)
mask_image_dir = history + '/{uuid}/mask_image'.format(uuid=uuid)
# mask反转
merge_after_mask_cut_image_dir = history + '/{uuid}/merge_after_mask_cut_image'.format(uuid=uuid)
controlnet_images = history + '/{uuid}/controlnet_images/{idx}'
template_test_images = history + '/template_test_images/'

# generate 的图片
generate_image_dir = history + '/{uuid}/generate_image/{idx}'
generate_glob_img = history + '/*/generate_image/*/*.png'
generate_self_innovate_image_dir = history + '/{uuid}/generate_self_innovate_image/{idx}'
generate_self_innovate_glob_img = history + '/*/generate_self_innovate_image/*/*.png'


dress_worker_history = dress_history + "/{worker_id}"
dress_worker_input_history = dress_history + "/{worker_id}/input"
dress_worker_output_history = dress_history + "/{worker_id}/output"
# clothes_merge_scene_dir = clothes_history + '/{uuid}/clothes_merge_scene_image'.format(uuid=uuid)
# clothes_dir = clothes_history + '/{uuid}/clothes_image'.format(uuid=uuid)
# clothes_mask_dir = clothes_history + '/{uuid}/clothes_mask_image'.format(uuid=uuid)
# clothes_mask_cut_dir = clothes_history + '/{uuid}/clothes_mask_cut_image'.format(uuid=uuid)
#
# clothes_generate_image_dir = clothes_history + '/{uuid}/clothes_generate_image/{idx}'
# clothes_generate_glob_img = clothes_history + '/*/clothes_generate_image/*/*.png'


# api 输入图片和输出图片
api_generate_commodity_dir = history + '/api_generate_commodity/{id_task}/{type}'
api_shadow_simple_gb_dir = history+"/api_shadow_simple_bg_img/{id_task}/{type}"

generate_inpaint_image_dir = history = f"{project_dir}/worker_data/inpaint_output/"
pathlib.Path(generate_inpaint_image_dir).mkdir(parents=True, exist_ok=True)


def get_file_idx(is_star=False, check_dir=commodity_merge_scene_image_dir) -> str:
    if not os.path.exists(check_dir):
        raise Exception("base dir is not exist")
    dirs = os.listdir(check_dir)
    if is_star:
        return str(len(dirs))
    if len(dirs) > 0:
        return dirs[len(dirs) - 1].split(".")[0]
    else:
        return '0'


def get_history_dirs():
    return os.listdir(history)

print(f"uuid:{uuid}")