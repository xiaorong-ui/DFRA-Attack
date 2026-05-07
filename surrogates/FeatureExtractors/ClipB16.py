import os
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
from torchvision import transforms
from torch.nn import functional as F
from .Base import BaseFeatureExtractor
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel

class ClipB16FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipB16FeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            attn_implementation="eager")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.normalizer = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features
    
    def text_features(self, texts: list[str]):
        text_inputs = self.processor(
            text=texts, return_tensors="pt",
            padding=True, truncation=True
        ).to(self.model.device)
        text_outputs = self.model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        return text_embeds

    def global_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)

        return global_feature

    def global_local_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=-1, keepdim=True)

        return global_feature, local_feature
    
    def global_local_features_visual(self, x):
        # x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        # image_features = self.model.get_image_features(**inputs)
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)

        inputs["pixel_values"] = inputs["pixel_values"]

        outputs = self.model.vision_model(pixel_values=inputs['pixel_values'], output_attentions=True)
        features = outputs.last_hidden_state
        attentions = outputs.attentions
        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=-1, keepdim=True)
        # features = self.model.get_image_embedding(inputs["pixel_values"])
        # global_feature = features[:, 0, :]
        # local_feature = features[:, 1:, :]
        
        visualize_feature(x, attentions[-1], save_path="visual/b16/b16_no_heatmap_last.png", alpha=0.5)
        for i in range(0, len(attentions)):
            visualize_feature(x, attentions[i], save_path=f"visual/b16/b16_no_heatmap_{i}.png", alpha=0.5)
        
        return global_feature, local_feature

    def global_local_middle_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True, output_attentions=True)
        features_7 = outputs.hidden_states[int(len(outputs.hidden_states)/2)]
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=-1, keepdim=True)

        local_features_middle = features_7[:, 1:, :]
        local_features_middle = local_features_middle / local_features_middle.norm(dim=-1, keepdim=True)
        global_feature_middle = features_7[:, 0, :]
        global_feature_middle = global_feature_middle / global_feature_middle.norm(dim=1, keepdim=True)
        return global_feature, local_feature, global_feature_middle, local_features_middle
    
    def global_middle_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
        features_7 = outputs.hidden_states[int(len(outputs.hidden_states)/2)] # 7 layers
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)

        global_feature_middle = features_7[:, 0, :]
        global_feature_middle = global_feature_middle / global_feature_middle.norm(dim=1, keepdim=True)

        return global_feature, global_feature_middle

    def global_index_features(self, x, i: float):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True)

        features = outputs.last_hidden_state
        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)

        i = int(i * len(outputs.hidden_states))  # scale float to index
        features_i = outputs.hidden_states[i]
        global_feature_i = features_i[:, 0, :]
        global_feature_i = global_feature_i / global_feature_i.norm(dim=1, keepdim=True)

        return global_feature, global_feature_i

    def global_local_index_features(self, x, i: float):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True)

        features_last = outputs.last_hidden_state
        global_feature_last = features_last[:, 0, :]
        global_feature_last = global_feature_last / global_feature_last.norm(dim=1, keepdim=True)
        local_feature_last = features_last[:, 1:, :]
        local_feature_last = local_feature_last / local_feature_last.norm(dim=-1, keepdim=True)

        i = int(i * len(outputs.hidden_states))
        features_i = outputs.hidden_states[i]
        global_feature_i = features_i[:, 0, :]
        global_feature_i = global_feature_i / global_feature_i.norm(dim=1, keepdim=True)
        local_features_i = features_i[:, 1:, :]
        local_features_i = local_features_i / local_features_i.norm(dim=-1, keepdim=True)

        return global_feature_last, local_feature_last, global_feature_i, local_features_i

def visualize_feature(
    image_batch,        # [B, 3, 224, 224]，数值在 0~1
    attention,          # [B, 12, 197, 197]
    index=0,
    alpha=0.7,
    save_path="feature_heatmap.png"
):
    img = image_batch[index]        # [3, 224, 224]

    attn_map = attention.mean(dim=1)[0, 0, 1:].reshape(-1, 1)      # [196, 1]
    # attn_map = attention.mean(dim=1)[0, 1:, 1:].mean(dim=0).reshape(-1, 1)      # [196, 1]

    patch_w, patch_h = 14, 14
    image_w, image_h = 224, 224
    num_patches = patch_w * patch_h
    attn_map = attn_map[:num_patches].reshape(1, 1, patch_w, patch_h)   # [1, 1, 14, 14]

    attn_map = F.interpolate(
        attn_map,
        size=(image_w, image_h),
        mode="bilinear",
        align_corners=False
    ).squeeze()             # [224, 224]

    attn_map = attn_map.detach().cpu()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    heat = attn_map.numpy()           # [224, 224]

    cmap = cm.get_cmap("jet")
    heatmap_rgba = cmap(heat)          # [224, 224, 4]
    heatmap_rgb = heatmap_rgba[..., :3]     # 去掉 alpha 通道，取 RGB，范围 0~1

    img_np = img.detach().cpu().permute(1, 2, 0).numpy() / 255.0
    img_np = img_np.clip(0.0, 1.0)

    overlay = (1 - alpha) * img_np + alpha * heatmap_rgb
    overlay = (overlay.clip(0.0, 1.0) * 255).astype(np.uint8)

    overlay_img = Image.fromarray(overlay)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    overlay_img.save(save_path)
    print(f"saved heatmap to: {save_path}")

class ClipB16FeatureExtractorOT(BaseFeatureExtractor):
    def __init__(self):
        super(ClipB16FeatureExtractorOT, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.normalizer = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        features = self.model.get_image_embedding(inputs["pixel_values"])
        global_feature = features[:,0,:]
        local_feature = features[:,1:,:]
        return global_feature, local_feature
    
    
class ClipB16FeatureExtractorOT_middle(BaseFeatureExtractor):
    def __init__(self):
        super(ClipB16FeatureExtractorOT, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.normalizer = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        outputs = self.model.get_image_embedding(pixel_values=inputs["pixel_values"], output_hidden_states=True, return_dict=True)
        features_7 = outputs.hidden_states[int(len(outputs.hidden_states)/2)]

        features = outputs.last_hidden_state
        global_feature = features[:, 0, :]
        local_feature = features[:, 1:, :]

        local_features_middle = features_7[:, 1:, :]
        local_features_middle = local_features_middle / local_features_middle.norm(dim=-1, keepdim=True)
        global_feature_middle = features_7[:, 0, :]
        global_feature_middle = global_feature_middle / global_feature_middle.norm(dim=1, keepdim=True)

        return global_feature, local_feature, global_feature_middle,local_features_middle
