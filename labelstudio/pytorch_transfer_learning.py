import hashlib
import io
import json
import logging
import os
import shutil
import time
import urllib

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from appdirs import user_cache_dir, user_data_dir
from urllib.parse import urlparse

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_choice,
    get_env,
    # get_local_path,
    get_single_tag_keys,
    is_skipped,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


HOSTNAME = get_env("HOSTNAME", "http://host.docker.internal:8080")
MODEL_DIR = os.getenv("MODEL_DIR")
API_KEY = get_env("API_KEY", "12345")

print("=> LABEL STUDIO HOSTNAME = ", HOSTNAME)
print("=> API_KEY = ", API_KEY)
if not API_KEY:
    print("=> WARNING! API_KEY is not set")

image_size = 224
image_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
image_cache_dir = os.path.join(os.path.dirname(__file__), "image-cache")
os.makedirs(image_cache_dir, exist_ok=True)

LOCAL_FILES_DOCUMENT_ROOT = get_env(
    'LOCAL_FILES_DOCUMENT_ROOT', default=os.path.abspath(os.sep)
)
_DIR_APP_NAME = 'label-studio'


logger = logging.getLogger(__name__)


def get_data_dir():
    data_dir = user_data_dir(appname=_DIR_APP_NAME)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_cache_dir():
    cache_dir = user_cache_dir(appname=_DIR_APP_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_local_path(
    url,
    cache_dir=None,
    project_dir=None,
    hostname=None,
    image_dir=None,
    access_token=None,
    download_resources=True,
):
    """
    Get local path for url
    :param url: File url
    :param cache_dir: Cache directory to download or copy files
    :param project_dir: Project directory
    :param hostname: Hostname for external resource
    :param image_dir: Image directory
    :param access_token: Access token for external resource (e.g. LS backend)
    :param download_resources: Download external files
    :return: filepath
    """
    is_local_file = url.startswith('/data/') and '?d=' in url
    is_uploaded_file = url.startswith('/data/upload')
    if image_dir is None:
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        image_dir = project_dir and os.path.join(project_dir, 'upload') or upload_dir

    # File reference created with --allow-serving-local-files option
    if is_local_file:
        filename, dir_path = url.split('/data/', 1)[-1].split('?d=')
        dir_path = str(urllib.parse.unquote(dir_path))
        filepath = os.path.join(LOCAL_FILES_DOCUMENT_ROOT, dir_path)
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        return filepath

    # File uploaded via import UI
    elif is_uploaded_file and os.path.exists(image_dir):
        project_id = url.split("/")[-2]  # To retrieve project_id
        image_dir = os.path.join(image_dir, project_id)
        filepath = os.path.join(image_dir, os.path.basename(url))
        if cache_dir and download_resources:
            shutil.copy(filepath, cache_dir)
        return filepath

    elif is_uploaded_file and hostname:
        url = hostname + url
        logger.info('Resolving url using hostname [' + hostname + '] from LSB: ' + url)

    elif is_uploaded_file:
        raise FileNotFoundError(
            "Can't resolve url, neither hostname or project_dir passed: " + url
        )

    if is_uploaded_file and not access_token:
        raise FileNotFoundError(
            "Can't access file, no access_token provided for Label Studio Backend"
        )

    # File specified by remote URL - download and cache it
    # print('reached: file specified by remote URL')
    cache_dir = cache_dir or get_cache_dir()
    parsed_url = urlparse(url)
    # print(f'parsed URL: {parsed_url}')
    url_filename = os.path.basename(parsed_url.path)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
    filepath = os.path.join(cache_dir, url_hash + '__' + url_filename)
    # print(f'filepath: {filepath}')
    if not os.path.exists(filepath):
        logger.info('Download {url} to {filepath}'.format(url=url, filepath=filepath))
        if download_resources:
            # print('reached: if download resources')
            # check if url matches hostname - then uses access token to this Label Studio instance
            if access_token and hostname and parsed_url.netloc == urlparse(hostname).netloc:
                headers = {'Authorization': 'Token ' + access_token}
            else:
                headers = {}
            r = requests.get(url, stream=True, headers=headers)
            r.raise_for_status()
            with io.open(filepath, mode='wb') as fout:
                fout.write(r.content)
    return filepath


def get_transformed_image(url):
    filepath = get_local_path(url, hostname=HOSTNAME, access_token=API_KEY)

    with open(filepath, mode="rb") as f:
        image = Image.open(f).convert("RGB")

    return image_transforms(image)


class ImageClassifierDataset(Dataset):
    def __init__(self, image_urls, image_classes):
        self.classes = list(set(image_classes))
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        self.images, self.labels = [], []
        for image_url, image_class in zip(image_urls, image_classes):
            try:
                image = get_transformed_image(image_url)
            except Exception as exc:
                print(exc)
                continue
            self.images.append(image)
            self.labels.append(self.class_to_label[image_class])

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


class ImageClassifier(object):
    def __init__(self, num_classes, freeze_extractor=False):
        self.model = models.resnet18(pretrained=True)
        if freeze_extractor:
            print("Transfer learning with a fixed ConvNet feature extractor")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("Transfer learning with a full ConvNet finetuning")

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.model = self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        if freeze_extractor:
            self.optimizer = optim.SGD(
                self.model.fc.parameters(), lr=0.001, momentum=0.9
            )
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, image_urls):
        images = torch.stack([get_transformed_image(url) for url in image_urls]).to(
            device
        )
        with torch.no_grad():
            return self.model(images).to(device).data.numpy()

    def train(self, dataloader, num_epochs=5):
        since = time.time()

        self.model.train()
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                self.scheduler.step(epoch)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print("Train Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

        print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        return self.model


class ImageClassifierAPI(LabelStudioMLBase):
    def __init__(self, freeze_extractor=False, **kwargs):
        super(ImageClassifierAPI, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "Choices", "Image"
        )
        self.freeze_extractor = freeze_extractor
        if self.train_output:
            self.classes = self.train_output["classes"]
            self.model = ImageClassifier(len(self.classes), freeze_extractor)
            self.model.load(self.train_output["model_path"])
        else:
            self.model = ImageClassifier(len(self.classes), freeze_extractor)

    def reset_model(self):
        self.model = ImageClassifier(len(self.classes), self.freeze_extractor)

    def predict(self, tasks, **kwargs):
        image_urls = [task["data"][self.value] for task in tasks]
        logits = self.model.predict(image_urls)
        predicted_label_indices = np.argmax(logits, axis=1)
        print(f"predicted_label_indices: {predicted_label_indices}")
        predicted_scores = logits[
            np.arange(len(predicted_label_indices)), predicted_label_indices
        ]
        print(f"predicted scores: {predicted_scores}")
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.classes[idx]
            # prediction result for the single task
            result = [
                {
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "choices",
                    "value": {"choices": [predicted_label]},
                }
            ]

            # expand predictions with their scores for all tasks
            predictions.append({"result": result, "score": float(score)})

        return predictions

    def _get_annotated_dataset(self, project_id):
        """Just for demo purposes: retrieve annotated data from Label Studio API"""
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(
            download_url, headers={"Authorization": f"Token {API_KEY}"}
        )
        if response.status_code != 200:
            raise Exception(
                f"Can't load task data using {download_url}, "
                f"response status_code = {response.status_code}"
            )
        return json.loads(response.content)

    def fit(self, event, data, batch_size=32, num_epochs=20, **kwargs):
        image_urls, image_classes = [], []
        print("Collecting annotations...")
        # aju temp
        print(data)

        # project_id = data['project']['id']
        # aju temp
        import re

        match = re.search(r"/data/models/(\d)+.*", data)  # sigh
        project_id = match.group(1)
        tasks = self._get_annotated_dataset(project_id)

        for task in tasks:
            image_urls.append(task["data"]["image"])

            if not task.get("annotations"):
                continue
            annotation = task["annotations"][0]
            # get input text from task data
            if annotation.get("skipped") or annotation.get("was_cancelled"):
                continue

            image_classes.append(annotation["result"][0]["value"]["choices"][0])

        print(f"Creating dataset with {len(image_urls)} images...")
        dataset = ImageClassifierDataset(image_urls, image_classes)
        print(f'got Dataset: {dataset}')
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        print("Train model...")
        self.reset_model()
        self.model.train(dataloader, num_epochs=num_epochs)

        print("Save model...")
        model_path = os.path.join(MODEL_DIR, "model.pt")
        self.model.save(model_path)
        print("Finish saving.")

        return {"model_path": model_path, "classes": dataset.classes}
