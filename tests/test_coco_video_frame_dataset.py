import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.io import write_png

    from evlearn.data.collate.jarr import (
        collate_batch_of_jagged_arrays_and_labels,
    )
    from evlearn.data.datasets.coco_video_frame_dataset import (
        COCOVideoFrameDataset,
    )
    from evlearn.data.samplers.jarr_subseq_sampler import VideoClipSampler

    HAS_DEPS = True
except ModuleNotFoundError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "torch/torchvision dependencies are unavailable")
class TestCOCOVideoFrameDataset(unittest.TestCase):

    def _write_image(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = torch.full((3, 240, 304), value, dtype=torch.uint8)
        write_png(image, str(path))

    def _make_dataset_root(self, tmpdir):
        root = Path(tmpdir)

        self._write_image(root / "images" / "test" / "v1_00000.png", 10)
        self._write_image(root / "images" / "test" / "v1_00001.png", 20)
        self._write_image(root / "images" / "test" / "v2_00000.png", 30)

        coco = {
            "videos": [
                {"id": 1, "name": "v1"},
                {"id": 2, "name": "v2"},
            ],
            "images": [
                {
                    "id": 2,
                    "video_id": 1,
                    "frame_id": 1,
                    "file_name": "images/test/v1_00001.png",
                    "width": 304,
                    "height": 240,
                    "timestamp_us": 100690,
                },
                {
                    "id": 1,
                    "video_id": 1,
                    "frame_id": 0,
                    "file_name": "images/test/v1_00000.png",
                    "width": 304,
                    "height": 240,
                    "timestamp_us": 50690,
                },
                {
                    "id": 3,
                    "video_id": 2,
                    "frame_id": 0,
                    "file_name": "images/test/v2_00000.png",
                    "width": 304,
                    "height": 240,
                    "timestamp_us": 150690,
                },
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 2,
                    "video_id": 1,
                    "frame_id": 1,
                    "category_id": 0,
                    "bbox": [229.0, 139.0, 61.0, 32.0],
                    "area": 1952.0,
                    "iscrowd": 0,
                    "instance_id": 434,
                },
                {
                    "id": 2,
                    "image_id": 3,
                    "video_id": 2,
                    "frame_id": 0,
                    "category_id": 1,
                    "bbox": [10.0, 20.0, 15.0, 25.0],
                    "area": 375.0,
                    "iscrowd": 0,
                    "instance_id": 9,
                },
            ],
            "categories": [
                {"id": 0, "name": "class_0"},
                {"id": 1, "name": "class_1"},
            ],
        }

        ann_file = root / "labels" / "test.json"
        ann_file.parent.mkdir(parents=True, exist_ok=True)
        ann_file.write_text(json.dumps(coco), encoding="utf-8")
        return root, ann_file

    def _make_dataset(self, root, ann_file):
        return COCOVideoFrameDataset(
            path=str(root),
            split="test",
            ann_files={"test": str(ann_file)},
            label_dtype_list=[
                ("boxes", None),
                ("labels", "int32"),
                ("psee_labels", None),
                ("area", None),
            ],
            bbox_fmt="xywh",
            canvas_size=[240, 304],
            return_index=True,
        )

    def test_xywh_annotations_are_returned_as_xyxy_boxes(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root, ann_file = self._make_dataset_root(tmpdir)
            dataset = self._make_dataset(root, ann_file)

            self.assertEqual(dataset.array_specs.get_n_arrays(), 2)
            self.assertEqual(dataset.array_specs.get_array_length(0), 2)
            self.assertEqual(dataset.array_specs.get_array_length(1), 1)

            _frame0, _index0, labels0 = dataset.get_elem(0, 0)
            self.assertIsNone(labels0)

            frame1, index1, labels1 = dataset.get_elem(0, 1)
            self.assertEqual(tuple(frame1.shape), (3, 240, 304))
            self.assertEqual(index1.tolist(), [0, 1])

            self.assertEqual(labels1["labels"].tolist(), [0])
            self.assertTrue(torch.equal(
                torch.as_tensor(labels1["boxes"]),
                torch.tensor([[229.0, 139.0, 290.0, 171.0]]),
            ))
            self.assertEqual(labels1["area"].tolist(), [1952.0])

            psee_labels = labels1["psee_labels"]
            self.assertEqual(int(psee_labels["t"][0]), 100690)
            self.assertEqual(float(psee_labels["x"][0]), 229.0)
            self.assertEqual(float(psee_labels["w"][0]), 61.0)
            self.assertEqual(int(psee_labels["class_id"][0]), 0)

    def test_video_clip_sampler_and_collate_reuse_existing_pipeline(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root, ann_file = self._make_dataset_root(tmpdir)
            dataset = self._make_dataset(root, ann_file)
            sampler = VideoClipSampler(
                dataset=dataset,
                batch_size=2,
                shuffle_videos=False,
                shuffle_frames=False,
                shuffle_clips=False,
                skip_unlabeled=False,
                split_by_video_starts=False,
                drop_last=False,
                pad_empty=True,
                clip_length=2,
                seed=0,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=lambda batch: collate_batch_of_jagged_arrays_and_labels(
                    batch,
                    batch_first=False,
                ),
            )

            frames, indices, labels = next(iter(loader))

            self.assertEqual(tuple(frames.shape), (2, 2, 3, 240, 304))
            self.assertEqual(tuple(indices.shape), (2, 2, 2))
            self.assertEqual(len(labels), 2)
            self.assertEqual(len(labels[0]), 2)


if __name__ == "__main__":
    unittest.main()
