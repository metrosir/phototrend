import sys
import os
from utils.utils import project_dir, models_path, paths
from PIL import Image
import torch
import scripts.errors as errors
import glob
from pathlib import Path
import re
from collections import namedtuple

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

Category = namedtuple("Category", ["name", "topn", "items"])
re_topn = re.compile(r"\.top(\d+)\.")
blip_image_eval_size = 384
clip_model_name = 'ViT-L/14'

interrogate_keep_models_in_memory=False

interrogate_clip_dict_limit=1500
'''
    CLIP: maximum number of lines in text file (0 = No limit)
'''
interrogate_clip_num_beams=1
'''
    Interrogate: num_beams for BLIP
'''
interrogate_clip_min_length=24
'''
    Interrogate: minimum description length (excluding artists, etc..)
'''

interrogate_clip_max_length=48
'''
    Interrogate: maximum description length
'''

interrogate_return_ranks=False
'''
    Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators).
'''

def download_default_clip_interrogate_categories(content_dir):
    print("Downloading CLIP categories...")

    tmpdir = content_dir + "_tmp"
    category_types = ["artists", "flavors", "mediums", "movements"]

    try:
        os.makedirs(tmpdir)
        for category_type in category_types:
            torch.hub.download_url_to_file(f"https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/{category_type}.txt", os.path.join(tmpdir, f"{category_type}.txt"))
        os.rename(tmpdir, content_dir)

    except Exception as e:
        errors.display(e, "downloading default CLIP interrogate categories")
    finally:
        if os.path.exists(tmpdir):
            os.remove(tmpdir)

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
class InterrogateModels:
    blip_model = None
    clip_model = None
    clip_preprocess = None
    dtype = None
    running_on_cpu = None

    def __init__(self):
        self.loaded_categories = None
        self.skip_categories = []
        self.content_dir = os.path.join(models_path, 'interrogate')
        self.running_on_cpu = False

    def categories(self):
        return []
        if not os.path.exists(self.content_dir):
            download_default_clip_interrogate_categories(self.content_dir)

        self.loaded_categories = []

        if os.path.exists(self.content_dir):
            self.skip_categories = []
            category_types = []

            for filename in Path(self.content_dir).glob('*.txt'):
                category_types.append(filename.stem)
                if filename.stem in self.skip_categories:
                    continue
                m = re_topn.search(filename.stem)
                topn = 1 if m is None else int(m.group(1))
                with open(filename, "r", encoding="utf8") as file:
                    lines = [x.strip() for x in file.readlines()]

                self.loaded_categories.append(Category(name=filename.stem, topn=topn, items=lines))

        return self.loaded_categories


    def create_fake_fairscale(self):
        class FakeFairscale:
            def checkpoint_wrapper(self):
                pass

        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = FakeFairscale

    def load_blip_model(self, name = "model_base_caption_capfilt_large.pth"):
        import models.blip

        model_dir = models_path+"/BLIP"
        remote_model_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/" + name
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=model_dir)
        blip_model = models.blip.blip_decoder(pretrained=f'{model_dir}/{name}', image_size=blip_image_eval_size, vit='base',
                                              med_config=os.path.join(paths["BLIP"], "configs", "med_config.json"))
        blip_model.eval()
        return blip_model

    def load_clip_model(self):
        import clip
        model, preprocess = clip.load(clip_model_name, download_root=os.path.join(models_path, "CLIP"))
        model.eval()
        model = model.to("cuda")
        return model, preprocess

    def load(self):
        print("self.blip_model:", self.blip_model)
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            self.blip_model = self.blip_model.half()
        self.blip_model = self.blip_model.to("cuda")

        print("self.clip_model:", self.clip_model)
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = self.load_clip_model()
            self.clip_model = self.clip_model.half()

        self.clip_model = self.clip_model.to("cuda")

        self.dtype = next(self.clip_model.parameters()).dtype

    def send_clip_to_ram(self):
        if not interrogate_keep_models_in_memory:
            if self.clip_model is not None:
                self.clip_model = self.clip_model.to(torch.device("cpu"))

    def send_blip_to_ram(self):
        if not interrogate_keep_models_in_memory:
            if self.blip_model is not None:
                self.blip_model = self.blip_model.to(torch.device("cpu"))

    def unload(self):
        self.send_clip_to_ram()
        self.send_blip_to_ram()

        torch_gc()

    def rank(self, image_features, text_array, top_count=1):
        import clip

        torch_gc()

        if interrogate_clip_dict_limit != 0:
            text_array = text_array[0:int(interrogate_clip_dict_limit)]

        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize([text for text in text_array], truncate=True).to("cuda")
        text_features = self.clip_model.encode_text(text_tokens).type(self.dtype)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = torch.zeros((1, len(text_array))).to("cuda")
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

    def generate_caption(self, pil_image):

        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to("cuda")

        with torch.no_grad():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=interrogate_clip_num_beams, min_length=interrogate_clip_min_length, max_length=interrogate_clip_max_length)

        return caption[0]

    def interrogate(self, pil_image):
        res = ""
        self.load()

        caption = self.generate_caption(pil_image)
        self.send_blip_to_ram()
        torch_gc()
        res = caption
        clip_image = self.clip_preprocess(pil_image).unsqueeze(0).type(self.dtype).to('cuda')
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = self.clip_model.encode_image(clip_image).type(self.dtype)

            image_features /= image_features.norm(dim=-1, keepdim=True)

            for name, topn, items in self.categories():
                matches = self.rank(image_features, items, top_count=topn)
                print("matches:", matches)
                for match, score in matches:
                    if interrogate_return_ranks:
                        res += f", ({match}:{score/100:.3f})"
                    else:
                        res += ", " + match
        self.unload()
        return res