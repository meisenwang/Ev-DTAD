"""
COCO-VID image dataset with the same jagged-array interface as EBC frames.

The dataset reads RGB images referenced by a COCO-style JSON file and exposes
each video as one jagged array so the existing video-clip sampler can be reused.
Input annotations use COCO bbox format [x, y, width, height] by default; labels
returned to the model use torchvision BoundingBoxes in XYXY format.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image

from .funcs import cantor_pairing, nan_sanitize
from .funcs_frame import apply_transforms_to_data
from .jagged_array_dataset import JaggedArrayDataset
from .jagged_array_specs import ElemSpec, JArrSpec, SimpleJaggedArraySpecs


PSEE_BBOX_DTYPE = np.dtype({
    'names'   : [
        't', 'x', 'y', 'w', 'h', 'class_id', 'track_id',
        'class_confidence',
    ],
    'formats' : ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
    'offsets' : [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize' : 40,
})


def _resolve_path(root, path):
    root = os.fspath(root)
    path = os.fspath(path)
    if os.path.isabs(path):
        return os.path.normpath(path)
    path = path.replace('\\', os.sep)
    return os.path.normpath(os.path.join(root, path))


def _torch_dtype(dtype):
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    return getattr(torch, str(dtype))


def _select_ann_file(ann_files, split):
    if isinstance(ann_files, str):
        return ann_files

    if split in ann_files:
        return ann_files[split]

    raise KeyError(
        f"No annotation file configured for split '{split}'."
        f" Available splits: {list(ann_files.keys())}."
    )


def _category_mapping(categories):
    cat_ids = sorted(int(cat['id']) for cat in categories)
    return {cat_id: idx for idx, cat_id in enumerate(cat_ids)}


def _image_sort_key(image):
    return (
        int(image.get('video_id', -1)),
        int(image.get('frame_id', image.get('id', 0))),
        int(image.get('id', 0)),
    )


def _xywh_to_xyxy(boxes):
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    result = boxes.copy()
    result[:, 2] = result[:, 0] + result[:, 2]
    result[:, 3] = result[:, 1] + result[:, 3]
    return result


def _sanitize_xyxy(boxes, width, height):
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)

    x0 = np.minimum(boxes[:, 0], boxes[:, 2])
    y0 = np.minimum(boxes[:, 1], boxes[:, 3])
    x1 = np.maximum(boxes[:, 0], boxes[:, 2])
    y1 = np.maximum(boxes[:, 1], boxes[:, 3])

    boxes[:, 0] = x0
    boxes[:, 1] = y0
    boxes[:, 2] = x1
    boxes[:, 3] = y1
    return boxes


class COCOVideoFrameDataset(JaggedArrayDataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path, split, ann_files,
        skip_unlabeled       = False,
        data_dtype_list      = [('frame', 'float32')],
        label_dtype_list     = [('boxes', 'float32'), ('labels', 'int32')],
        transform_video      = None,
        transform_frame      = None,
        transform_labels     = None,
        bbox_fmt             = 'xywh',
        canvas_size          = None,
        return_vdir_fname    = False,
        return_index         = False,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=dangerous-default-value
        self._root = path
        self._split = split
        self._ann_file = _resolve_path(path, _select_ann_file(ann_files, split))
        self._data_dtype_list = data_dtype_list
        self._label_dtype_list = label_dtype_list
        self._label_names = [name for (name, _dtype) in label_dtype_list]

        self._transform_video = transform_video
        self._transform_frame = transform_frame
        self._transform_labels = transform_labels

        self._bbox_fmt = bbox_fmt.lower()
        if self._bbox_fmt not in ('xywh', 'xyxy'):
            raise ValueError(
                "COCOVideoFrameDataset supports bbox_fmt 'xywh' or 'xyxy',"
                f" got '{bbox_fmt}'."
            )

        self._canvas_size = canvas_size
        self._return_vdir_fname = return_vdir_fname
        self._return_index = return_index

        with open(self._ann_file, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        self._cat_id_to_label = _category_mapping(coco.get('categories', []))
        self._images = {
            int(image['id']): image for image in coco.get('images', [])
        }
        self._anns_by_image = self._group_annotations(coco.get('annotations', []))
        self._image_to_video_name = self._build_video_names(coco.get('videos', []))

        video_specs = self._build_video_specs(skip_unlabeled)
        super().__init__(SimpleJaggedArraySpecs(video_specs))

    def _group_annotations(self, annotations):
        result = defaultdict(list)

        for ann in annotations:
            if int(ann.get('iscrowd', 0)) != 0:
                continue

            result[int(ann['image_id'])].append(ann)

        return result

    def _build_video_names(self, videos):
        video_names = {}

        for video in videos:
            video_id = int(video['id'])
            video_names[video_id] = video.get('name', str(video_id))

        return video_names

    def _build_video_specs(self, skip_unlabeled):
        grouped = defaultdict(list)

        for image in sorted(self._images.values(), key=_image_sort_key):
            image_id = int(image['id'])
            video_id = int(image.get('video_id', image_id))
            grouped[video_id].append(image)

        result = []
        for video_id in sorted(grouped):
            elems = []
            for image in grouped[video_id]:
                image_id = int(image['id'])
                anns = self._anns_by_image.get(image_id, [])
                if skip_unlabeled and not anns:
                    continue

                elems.append(ElemSpec(
                    int(image.get('frame_id', image_id)),
                    image_id,
                    image_id if anns else None,
                ))

            if elems:
                video_name = self._image_to_video_name.get(video_id, str(video_id))
                result.append(JArrSpec(video_name, elems))

        return result

    def get_null_elem(self):
        result = tuple(None for _ in range(len(self._data_dtype_list) + 1))

        if self._return_index:
            result += (None,)

        return result

    def get_video_seed(self, arr_idx):
        return cantor_pairing(self._seed, arr_idx)

    def _load_frame(self, image):
        path = _resolve_path(self._root, image['file_name'])
        frame = read_image(path, mode=ImageReadMode.RGB)

        if len(self._data_dtype_list) != 1:
            raise ValueError(
                "COCOVideoFrameDataset currently returns one data tensor named"
                " 'frame'."
            )

        data_name, dtype = self._data_dtype_list[0]
        if data_name != 'frame':
            raise ValueError(
                "COCOVideoFrameDataset expects data_dtype_list to request"
                f" 'frame', got '{data_name}'."
            )

        dtype = _torch_dtype(dtype)
        if dtype is not None:
            frame = frame.to(dtype=dtype)

        return frame

    def _make_label_arrays(self, image, anns):
        width = int(image['width'])
        height = int(image['height'])

        boxes_in = np.asarray(
            [ann['bbox'] for ann in anns], dtype=np.float32
        ).reshape(-1, 4)

        if self._bbox_fmt == 'xywh':
            boxes_xywh = boxes_in.copy()
            boxes_xyxy = _xywh_to_xyxy(boxes_in)
        else:
            boxes_xyxy = boxes_in.copy()
            boxes_xywh = boxes_in.copy()
            boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
            boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]

        boxes_xyxy = _sanitize_xyxy(boxes_xyxy, width, height)
        keep = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) \
            & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])

        boxes_xyxy = boxes_xyxy[keep]
        boxes_xywh = boxes_xywh[keep]
        kept_anns = [ann for (ann, k) in zip(anns, keep) if k]

        labels = np.asarray([
            self._cat_id_to_label.get(
                int(ann['category_id']), int(ann['category_id'])
            ) for ann in kept_anns
        ], dtype=np.int64)

        return boxes_xyxy, boxes_xywh, labels, kept_anns

    def _make_psee_labels(self, image, boxes_xywh, labels, anns):
        timestamp = int(image.get(
            'timestamp_us', image.get('frame_id', image.get('id', 0))
        ))

        result = np.zeros((len(anns),), dtype=PSEE_BBOX_DTYPE)
        if len(anns) == 0:
            return result

        result['t'] = timestamp
        result['x'] = boxes_xywh[:, 0]
        result['y'] = boxes_xywh[:, 1]
        result['w'] = boxes_xywh[:, 2]
        result['h'] = boxes_xywh[:, 3]
        result['class_id'] = labels.astype(np.uint32)
        result['track_id'] = np.asarray([
            max(0, int(ann.get('instance_id', ann.get('track_id', 0))))
            for ann in anns
        ], dtype=np.uint32)
        result['class_confidence'] = 1.0
        return result

    def _load_labels(self, image):
        anns = self._anns_by_image.get(int(image['id']), [])
        if not anns:
            return None

        boxes_xyxy, boxes_xywh, labels_np, kept_anns = \
            self._make_label_arrays(image, anns)

        if len(kept_anns) == 0:
            return None

        height = int(image['height'])
        width = int(image['width'])
        canvas_size = self._canvas_size or (height, width)

        labels = {}
        if 'boxes' in self._label_names:
            labels['boxes'] = tv_tensors.BoundingBoxes(
                boxes_xyxy,
                format='XYXY',
                canvas_size=canvas_size,
            )

        if 'labels' in self._label_names:
            labels['labels'] = torch.from_numpy(labels_np)

        if 'psee_labels' in self._label_names:
            labels['psee_labels'] = self._make_psee_labels(
                image, boxes_xywh, labels_np, kept_anns
            )

        if 'area' in self._label_names:
            labels['area'] = torch.as_tensor([
                float(ann.get('area', boxes_xywh[idx, 2] * boxes_xywh[idx, 3]))
                for idx, ann in enumerate(kept_anns)
            ], dtype=torch.float32)

        if 'iscrowd' in self._label_names:
            labels['iscrowd'] = torch.as_tensor([
                int(ann.get('iscrowd', 0)) for ann in kept_anns
            ], dtype=torch.uint8)

        return labels

    def get_elem(self, arr_idx, elem_idx):
        vdir = self._specs.get_array_name(arr_idx)
        (_frame_idx, image_id, _labels_id) = \
            self._specs.get_elem_spec(arr_idx, elem_idx)
        image = self._images[int(image_id)]

        data = [self._load_frame(image)]
        labels = self._load_labels(image)

        nan_sanitize(data[0], f"Raw data has NaNs: File: {image['file_name']}")

        data, labels = apply_transforms_to_data(
            data, labels, self._transform_video, self._transform_frame,
            self._transform_labels, squash_time_polarity=False,
            video_seed=self.get_video_seed(arr_idx)
        )

        nan_sanitize(
            data[0],
            f"Data after transforms has NaN: File: {image['file_name']}"
        )

        if self._return_vdir_fname:
            if labels is None:
                labels = {}

            labels['vdir'] = vdir
            labels['fname_data'] = image['file_name']
            labels['fname_labels'] = self._ann_file

        if self._return_index:
            data.append(
                torch.Tensor([arr_idx, elem_idx]).to(dtype=torch.int32)
            )

        return (*data, labels)
