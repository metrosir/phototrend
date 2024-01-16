

class DressPipe:
    def __init__(self, model_img, scene_img, dress_mask_img, base_mode_id):
        self._pipe = None

    def setControlNet(self):
        pass

    def __call__(self,
                 prompt,
                 negative_prompt,
                 height,
                 weight,
                 steps,
                 sampler_name,
                 strength,
                 open_after,
                 after_params,
                 res_img_info,
                 use_ip_adapter,
                 ipadapter_img):
        pass
