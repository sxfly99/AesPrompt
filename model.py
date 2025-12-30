import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

_tokenizer = _Tokenizer()

class ITC_model(nn.Module):
    def __init__(self, clip_name):
        super(ITC_model, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_size = self.select_clip(clip_name)
        self.clip_model = clip_model.float().to(self.device)
        self.clip_size = clip_size['feature_size']

    def select_clip(self, clip_name):
        param = {'feature_size': 512}
        if clip_name == 'RN50':
            clip_model, _ = clip.load("RN50", device=self.device)
            param['feature_size'] = 1024
        elif clip_name == 'ViT-B/16':
            clip_model, _ = clip.load("ViT-B/16", device=self.device)
            param['feature_size'] = 768
        else:
            raise IOError('model type is wrong')

        # Set the device attribute for the clip_model
        clip_model.device = self.device

        return clip_model, param

    def forward(self, x, texts):
        img_embedding = self.clip_model.visual(x.to(self.device))
        try:
            text_tokens = torch.cat([clip.tokenize(text) for text in texts])
            text_embedding = self.clip_model.encode_text(text_tokens.to(self.device)).float()
            return img_embedding, text_embedding
        except:
            print('Error', texts)
            return img_embedding, img_embedding

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype).to(prompts.device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class Scene_Model_SPAQ(nn.Module):
    def __init__(self, num_classes):
        super(Scene_Model_SPAQ, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MobileNetV3-Large model with weights
        mobilenet_v3_large = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).to(device)

        # Remove the last classifier layer
        self.features = mobilenet_v3_large.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Define a new classifier
        self.mlp = nn.Sequential(
            nn.Linear(960, num_classes)  # 1280 is the output feature dimension of MobileNetV3-Large
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        # output = self.mlp(x)
        return x

class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        clip_model, _ = clip.load("RN50", device=device)

        self.clip_model = clip_model.float() # Using ResNet50 as a placeholder for CLIP-RN50
        self.mlp = nn.Sequential(
            nn.Linear(2048, 8)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        emotion_features = self.clip_model.visual(x).float()
        emotion_features = self.avgpool(emotion_features).view(emotion_features.size(0), -1)
        # output = self.mlp(emotion_features)

        return emotion_features


import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import RobertaTokenizer, RobertaModel


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('./pretrained_ckpt/roberta-base')
        self.bert_model = RobertaModel.from_pretrained('./pretrained_ckpt/roberta-base')
        self.aes_dim = 768
        # loading sem model
        sem_model = Scene_Model_SPAQ(num_classes=9)
        print('Loading spaq model:', sem_model.load_state_dict(
            torch.load('./pretrained_ckpt/best_SPAQ_model_mv3.pth')))
        self.sem_model = sem_model.to(clip_model.device)
        self.sem_dim = 960

        # loading emo model
        emo_model = EmotionClassifier()
        print('Loading emo model:',
              emo_model.load_state_dict(torch.load('./pretrained_ckpt/best_emotion_classifier.pth')))
        self.emo_model = emo_model.to(clip_model.device)
        self.emo_model.clip_model.visual.proj = None
        self.emo_dim = 2048

        self.device = clip_model.device
        n_cls = len(classnames)
        n_ctx = cfg["TRAINER"]["COCOOP"]["N_CTX"]
        ctx_init = cfg["TRAINER"]["COCOOP"]["CTX_INIT"]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg["INPUT"]["SIZE"][0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype).to(self.device)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=self.device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        self.sem_emo_adapter = nn.Linear(self.sem_dim + self.emo_dim, self.sem_dim)
        self.bias_adapter = nn.Linear(self.sem_dim, ctx_dim)
        self.aes_adapter = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(self.aes_dim, ctx_dim))
            # ("relu", nn.ReLU(inplace=True)),
            # ("linear2", nn.Linear(ctx_dim, ctx_dim))
        ])).to(self.device)

        self.cross_attn = nn.MultiheadAttention(embed_dim=ctx_dim, num_heads=4, batch_first=True)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(torch.long).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(self.device)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts
    def get_caps_feats(self, caps):
        input_ids = []
        attention_masks = []
        for text in caps:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt',
                return_attention_mask=True,
                truncation=True
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to('cuda')
        attention_masks = torch.cat(attention_masks, dim=0).to('cuda')
        outputs = self.bert_model(input_ids, attention_mask=attention_masks)
        pooled_output = outputs.pooler_output
        return pooled_output

    def forward(self, im_features, images, caps):
        # 提取sem特征
        with torch.no_grad():
            sem_feats = self.sem_model(images).squeeze()
            emo_feats = self.emo_model(images)
            aes_feats = self.get_caps_feats(caps)

        # 归一化特征
        sem_feats = nn.functional.normalize(sem_feats, p=2, dim=1)
        emo_feats = nn.functional.normalize(emo_feats, p=2, dim=1)

        sem_emo_feats = torch.cat([sem_feats, emo_feats], dim=1)
        sem_emo_feats = self.sem_emo_adapter(sem_emo_feats)
        aes_feats = self.aes_adapter(aes_feats)
        bias = sem_emo_feats + sem_feats
        bias = self.bias_adapter(bias)

        bias, _ = self.cross_attn(bias, aes_feats, aes_feats)
        bias = bias + aes_feats

        prefix = self.token_prefix.to(im_features.device)
        suffix = self.token_suffix.to(im_features.device)
        ctx = self.ctx.to(im_features.device)


        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias


        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class AesPrompt(nn.Module):
    def __init__(self, itc_model):
        super(AesPrompt, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = itc_model.clip_model.float().to(self.device)
        self.dtype = self.clip_model.dtype
        self.image_encoder = self.clip_model.visual.to(self.device)
        cfg = {
            "TRAINER": {
                "COCOOP": {
                    "N_CTX": 16,
                    "CTX_INIT": "",
                    "PREC": "fp32"
                }
            },
            "INPUT": {
                "SIZE": [224, 224]
            }
        }
        classnames = ['Good image', 'Average image', 'Bad image']

        self.prompt_learner = PromptLearner(cfg, classnames, self.clip_model).to(self.device)
        self.text_encoder = TextEncoder(self.clip_model).to(self.device)
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts.to(self.device)

    def forward(self, x, caps):
        final_scores = []
        img_embedding = self.image_encoder(x.type(self.dtype).to(self.device))
        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(img_embedding, x, caps)

        logits = []
        for pts_i, imf_i in zip(prompts, img_embedding):
            text_embedding = self.text_encoder(pts_i, self.tokenized_prompts)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            similarity = (100.0 * imf_i @ text_embedding.T).softmax(dim=-1)
            logits.append(similarity)

        logits_tensor = torch.stack(logits)

        for item in logits:
            good_score = item[0]
            average_score = item[1]
            bad_score = item[2]

            final_score = 1.0 * good_score + 0.5 * average_score + 0.0 * bad_score
            final_scores.append(final_score)

        return final_scores, logits_tensor