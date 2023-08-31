import io
import os

import albumentations as A
import clip
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, default_collate


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def random_nms_transform(detected_map):
    detected_map = cv2.resize(detected_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    detected_map = nms(detected_map, 127, 3.0)
    detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
    detected_map[detected_map > 4] = 255
    detected_map[detected_map < 255] = 0
    return detected_map


class ScribbleV2Preprocessor:
    def __init__(self):
        pass

    def HWC3(self, x: np.uint8):
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y

    def resize_image(self, input_image, resolution: int = 512):
        H, W, C = input_image.shape
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img

    def __call__(self, image: Image.Image, thr_a: int = 32) -> Image.Image:
        np_array = np.asarray(image, dtype=np.uint8)
        img = self.resize_image(self.HWC3(np_array), 512)
        g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
        g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
        dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
        result = np.zeros_like(img, dtype=np.uint8)
        result[2 * (255 - dog) > thr_a] = 255
        return Image.fromarray(result, "RGB")


####################################################
# DATALOADING AUGMENTATIONS AND PREPROCESSING
####################################################
aug_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(512, 512, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.CenterCrop(512, 512, p=1.0),
    ]
)
dropout_aug_pieline = A.Compose(
    [
        A.OneOf(
            [
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0),
                A.GridDropout(p=1.0),
                A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=1.0),
            ],
            p=0.5,
        ),
    ]
)
to_tensor = T.ToTensor()
normalize = T.Normalize([0.5], [0.5])
scribble_v2_preprocessor = ScribbleV2Preprocessor()
####################################################


def create_webdataset(
    urls,
    tokenizer,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""

    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.warn_and_continue,
    )
    dataset = dataset.shuffle(1000)

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}

        if enable_metadata:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            output["metadata"] = metadata

        if enable_image:
            image_data = item[image_key]
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # ----- base image augmentation -----
            image_array = np.asanyarray(image)
            image_array = aug_pipeline(image=image_array)["image"]
            image_array = Image.fromarray(image_array)
            # -----------------------------------

            # ------ scribble specific augmentation ------
            # randomly choose threshold for scribble_v2_preprocessor, between 1 and 40
            thr_a = np.random.randint(1, 40)
            scribble = scribble_v2_preprocessor(image_array, thr_a=thr_a)
            scribble = np.asanyarray(scribble)
            scribble = random_nms_transform(scribble)
            scribble = dropout_aug_pieline(image=scribble)["image"]
            # -------------------------------------------

            scribble = cv2.cvtColor(scribble, cv2.COLOR_RGB2GRAY)
            scribble = scribble.astype(np.float32) / 255.0
            scribble = torch.from_numpy(scribble).unsqueeze(0)

            image_tensor = to_tensor(image_array)
            image_tensor = normalize(image_tensor)

            if enable_metadata:
                output["metadata"]["image_filename"] = item["__key__"]

            output["pixel_values"] = image_tensor
            output["conditioning_pixel_values"] = scribble

        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = tokenizer(
                [caption],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]

            output["input_ids"] = tokenized_text

            if enable_metadata:
                output["metadata"]["text"] = caption

        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    # data = wds.WebLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_prepro_workers,
    #     pin_memory=True,
    #     prefetch_factor=2,
    #     collate_fn=collate_fn if input_format == "files" else None,
    # )

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    return data


class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        tokenizer,
        input_dataset,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        cache_path=None,
    ):
        self.batch_size = batch_size
        dataset = create_webdataset(
            input_dataset,
            tokenizer,
            enable_text=enable_text,
            enable_image=enable_image,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "webdataset")

    def __iter__(self):
        for batch in self.dataloader:
            yield batch


def create_webdataset_reader(
    tokenizer,
    input_dataset,
    batch_size,
    num_prepro_workers,
    enable_text=True,
    enable_image=True,
    enable_metadata=False,
    wds_image_key="jpg",
    wds_caption_key="txt",
    cache_path=None,
):
    return WebdatasetReader(
        tokenizer,
        input_dataset,
        batch_size,
        num_prepro_workers,
        enable_text=enable_text,
        enable_image=enable_image,
        enable_metadata=enable_metadata,
        wds_image_key=wds_image_key,
        wds_caption_key=wds_caption_key,
        cache_path=cache_path,
    )


# for i in [1, 2, 3, 4, 5, 6]:
#     image = Image.open(f"/mnt/data1/ayushman/projects/t2i_adapters/scripts/condition_images/NEW-SB-{i}.png")
#     processed = scribble_v2_preprocessor(image)
#     processed.save(f"/mnt/data1/ayushman/projects/t2i_adapters/scripts/condition_images/PROCESSED-NEW-SB-{i}.png")
