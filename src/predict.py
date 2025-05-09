import os
import pickle

import cv2
import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

os.environ["FAISS_OPT_LEVEL"] = ""


class Searcher:
    def __init__(self):
        self.dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dinov2_vits14.to(self.device)
        self.data_index = faiss.read_index("data/data.bin")
        with open("data/image_paths.pkl", "rb") as f:
            self.image_paths = pickle.load(f)

    def search_index(self, embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        embedding = np.array(embedding).reshape(1, -1)
        embedding = np.float32(embedding)
        faiss.normalize_L2(embedding)
        D, I = self.data_index.search(embedding, k)
        return I, D

    def search_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        embedding = self.dinov2_vits14(self.load_image(image).to(self.device))
        embedding = embedding[0].cpu().detach().numpy().reshape(1, -1)
        indices, scores = self.search_index(embedding)
        return indices, scores

    def load_image(self, image: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(image)
        transformed_img = transform_image(img)[:3].unsqueeze(0)
        return transformed_img

    def predict(self, image: np.ndarray) -> str:
        indices, scores = self.search_image(image)
        image_names = [self.image_paths[i] for i in indices[0]]
        result = {image_name: score for image_name, score in zip(image_names, scores[0].tolist())}
        return result
