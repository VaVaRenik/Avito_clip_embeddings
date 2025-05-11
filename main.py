import json
import os
import zipfile
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from PIL import Image

import torch
import clip
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from kaggle.api.kaggle_api_extended import KaggleApi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_image_path(image_id, root_dirs, glob_pattern="train_images_part_"):
    for root_dir in root_dirs:
        for chunk_dir in Path(root_dir).glob(f"{glob_pattern}*"):
            image_path = chunk_dir / f"{image_id}.jpg"
            if image_path.exists():
                return str(image_path)
    return None


class CLIPEmbedder:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.embedding_dim = self.model.visual.output_dim
        self.max_length = 77

    def get_text_embedding(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=1).cpu().numpy()
        return text_features

    def get_image_embedding(self, images):
        # images: list of PIL Images
        processed = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed)
            image_features = F.normalize(image_features, dim=1).cpu().numpy()
        return image_features


def encode_text(texts, embedder, batchsize=32):
    embeddings = []
    text_list = texts.tolist() if isinstance(texts, pd.Series) else texts

    for i in range(0, len(text_list), batchsize):
        batch = text_list[i:i + batchsize]
        emb = embedder.get_text_embedding(batch)
        embeddings.extend(emb)

    return np.array(embeddings)


def encode_image(ids, root_dirs, embedder, batchsize=32, glob_pattern='train_images_part_'):
    num_ids = len(ids)
    embeddings = np.zeros((num_ids, embedder.embedding_dim), dtype=np.float32)
    ids = ids.tolist() if isinstance(ids, pd.Series) else ids

    cnt_broken = 0
    for batch_start in range(0, num_ids, batchsize):
        batch_ids = ids[batch_start:batch_start + batchsize]
        batch_images = []
        valid_indices = []

        for i, image_id in enumerate(batch_ids):
            path = find_image_path(image_id, root_dirs, glob_pattern)
            if path:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    cnt_broken += 1

        if valid_indices:
            try:
                batch_embeddings = embedder.get_image_embedding(batch_images)

                for idx, emb in zip(valid_indices, batch_embeddings):
                    embeddings[batch_start + idx] = emb
            except Exception as e:
                print(f"Error processing batch: {str(e)}")

    print(f'Процент битых картинок: {cnt_broken / num_ids * 100: .2f}%')
    return embeddings


def batched_cosine_sim(base, cand, batch_size=1024):
    base_tensor = torch.tensor(base, dtype=torch.float32).to(device)
    cand_tensor = torch.tensor(cand, dtype=torch.float32).to(device)

    sims = []

    for i in range(0, len(base_tensor), batch_size):
        b = base_tensor[i:i + batch_size]
        c = cand_tensor[i:i + batch_size]

        b_norm = F.normalize(b, dim=1)
        c_norm = F.normalize(c, dim=1)

        sim = (b_norm * c_norm).sum(dim=1)
        sims.append(sim.cpu().numpy())

    return np.concatenate(sims)


class EmbeddingBuilder:
    def __init__(self, df, text_embedder, image_embedder, root_dirs, glob_pattern='train_images_part_'):

        emb_columns = [
            [
                'base_title',
                'base_description',
                [
                    'base_category_name',
                    'base_subcategory_name',
                    'base_param1',
                    'base_param2'
                ]
            ],
            [
                'cand_title',
                'cand_description',
                [
                    'cand_category_name',
                    'cand_subcategory_name',
                    'cand_param1',
                    'cand_param2'
                ]
            ]
        ]

        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.df = df
        self.output = None
        self.emb_columns = emb_columns
        self.root_dirs = root_dirs
        self.glob_pattern = glob_pattern

    def extract_text_embed(self):
        base_cols, cand_cols = self.emb_columns

        # Для base
        base_title_emb = encode_text(self.df[base_cols[0]], self.text_embedder)
        base_desc_emb = encode_text(self.df[base_cols[1]], self.text_embedder)

        combined_base = self.df[base_cols[2]].apply(
            lambda x: ' '.join(x.dropna().astype(str)),
            axis=1
        )
        base_cat_params_emb = encode_text(combined_base, self.text_embedder)

        # Для candidate
        cand_title_emb = encode_text(self.df[cand_cols[0]], self.text_embedder)
        cand_desc_emb = encode_text(self.df[cand_cols[1]], self.text_embedder)

        combined_cand = self.df[cand_cols[2]].apply(
            lambda x: ' '.join(x.dropna().astype(str)),
            axis=1
        )
        cand_cat_params_emb = encode_text(combined_cand, self.text_embedder)

        self.text_embeddings = {
            'base_title': base_title_emb,
            'base_description': base_desc_emb,
            'base_cat_params': base_cat_params_emb,
            'cand_title': cand_title_emb,
            'cand_description': cand_desc_emb,
            'cand_cat_params': cand_cat_params_emb
        }

    def extract_image_embed(self):
        base_image_ids = self.df['base_title_image'].tolist()
        cand_image_ids = self.df['cand_title_image'].tolist()

        base_image_emb = encode_image(
            base_image_ids,
            self.root_dirs,
            self.image_embedder,
            glob_pattern=self.glob_pattern
        )

        cand_image_emb = encode_image(
            cand_image_ids,
            self.root_dirs,
            self.image_embedder,
            glob_pattern=self.glob_pattern
        )

        self.image_embeddings = {
            'base_image': base_image_emb,
            'cand_image': cand_image_emb
        }

    def extract_cosine_sim(self):
        text_sims = []
        base_text_embs = []
        cand_text_embs = []

        # Собираем все текстовые косинусные вектора
        for key in self.text_embeddings:
            if key.startswith('base_'):
                suffix = key[len('base_'):]

                base_emb = self.text_embeddings[f'base_{suffix}']
                base_text_embs.append(base_emb)

                cand_emb = self.text_embeddings[f'cand_{suffix}']
                cand_text_embs.append(cand_emb)

                sim = batched_cosine_sim(base_emb, cand_emb)
                text_sims.append(sim)


        # Объединяем все текстовые вектора
        text_sims = np.stack(text_sims, axis=1)
        base_text_embs = np.stack(base_text_embs, axis=1)
        cand_text_embs = np.stack(cand_text_embs, axis=1)

        # Считаем mean и std
        base_text_mean = base_text_embs.mean(axis=1, keepdims=True)
        cand_text_mean = cand_text_embs.mean(axis=1, keepdims=True)

        base_text_std = base_text_embs.std(axis=1, keepdims=True)
        cand_text_std = cand_text_embs.std(axis=1, keepdims=True)

        # Обрабатываем image embeddings
        base_img_emb = self.image_embeddings.get('base_image')
        cand_img_emb = self.image_embeddings.get('cand_image')

        image_sim = batched_cosine_sim(base_img_emb, cand_img_emb)
        image_sim = image_sim.reshape(-1, 1)
        base_img_mean = base_img_emb.mean(axis=1, keepdims=True)
        cand_img_mean = cand_img_emb.mean(axis=1, keepdims=True)
        base_img_std = base_img_emb.std(axis=1, keepdims=True)
        cand_img_std = cand_img_emb.std(axis=1, keepdims=True)

        sim_matrix = np.hstack([
            text_sims,
            image_sim,
            base_text_mean,
            cand_text_mean,
            base_text_std,
            cand_text_std,
            base_img_mean,
            cand_img_mean,
            base_img_std,
            cand_img_std
        ])

        return sim_matrix.astype(np.float32)


def process_df(df, root_dirs, glob_pattern):
    text_embedder = CLIPEmbedder(device='cuda')
    image_embedder = text_embedder  # Один объект CLIP достаточно, т.к. модель одна

    builder = EmbeddingBuilder(
        df=df,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        root_dirs=root_dirs,
        glob_pattern=glob_pattern
    )

    builder.extract_text_embed()
    builder.extract_image_embed()
    output = builder.extract_cosine_sim()

    return output


train_df_paths = [
        "/app/data/tables/train_part_0001.snappy.parquet",
        # "/app/data/avito-tables/train_part_0002.snappy.parquet",
        # "/app/data/avito-tables/train_part_0003.snappy.parquet",
        # "/app/data/avito-tables/train_part_0004.snappy.parquet"
    ]
train_root_dirs = [
    "/app/data/train_images"
]

test_df_paths = [
    "/app/data/avito-tables/test_part_0001.snappy.parquet",
    # "/app/data/avito-tables/test_part_0002.snappy.parquet"
]
test_root_dirs = [
    "/app/data/test_images"
]

if __name__ == "__main__":
    for i, path in enumerate(train_df_paths, 1):
        df = pd.read_parquet(path)
        output = process_df(df, root_dirs=train_root_dirs, glob_pattern="train_images_part_")

        output_path = f"/app/output/train_part_{i}.npz"
        np.savez_compressed(output_path, output)

    # for i, path in enumerate(test_df_paths, 1):
    #     df = pd.read_parquet(path)
    #     output = process_df(df, root_dirs=test_root_dirs, glob_pattern="part_")

    #     output_path = f"/app/output/test_part_{i}.npz"
    #     np.savez_compressed(output_path, output)

    api = KaggleApi()
    api.authenticate()

    with zipfile.ZipFile('/app/output/results.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk('/app/output'):
            for file in files:
                if file.endswith('.npz'):
                    zipf.write(os.path.join(root, file), arcname=file)

    metadata = {
        "title": "Processed Avito Data",
        "id": "vavarenikk/avito-processed-data",
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open('/app/output/dataset-metadata.json', 'w') as f:
        json.dump(metadata, f)

    api.dataset_create_version(
        folder="/app/output",
        version_notes=f"Processed data from {len(train_df_paths)} train and {len(test_df_paths)} test files",
        quiet=False
    )