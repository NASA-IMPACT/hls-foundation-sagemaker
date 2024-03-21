import matplotlib.pyplot as plt
import mmcv
import torch

from mmseg.datasets.pipelines import Compose

from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
from .downloader import DOWNLOAD_FOLDER


# class LoadImage:
#     """A simple pipeline to load image."""

#     def __call__(self, results):
#         """Call function to load images into results.

#         Args:
#             results (dict): A result dict contains the file name
#                 of the image to be read.

#         Returns:
#             dict: ``results`` will be returned containing loaded image.
#         """

#         if isinstance(results["img"], str):
#             results["filename"] = results["img"]
#             results["ori_filename"] = results["img"]
#         else:
#             results["filename"] = None
#             results["ori_filename"] = None
#         img = mmcv.imread(results["img"])
#         results["img"] = img
#         results["img_shape"] = img.shape
#         results["ori_shape"] = img.shape
#         return results


class Infer:
    def __init__(self, config, checkpoint):
        self.config_filename = config
        self.checkpoint_filename = checkpoint

    def load_model(self):
        self.config = mmcv.Config.fromfile(self.config_filename)
        self.config.model.pretrained = None
        self.config.model.train_cfg = None

        if self.checkpoint_filename is not None:
            self.model = init_segmentor(self.config, self.checkpoint_filename, device="cuda:0")
            self.checkpoint = load_checkpoint(
                self.model, self.checkpoint_filename, map_location="cpu"
            )
            self.model.CLASSES = self.checkpoint["meta"]["CLASSES"]
            self.model.PALETTE = self.checkpoint["meta"]["PALETTE"]
        self.model.cfg = self.config  # save the config in the model for convenience
        self.model.to("cuda:0")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        return self.model

    def infer(self, images):
        """
        Infer on provided images
        Args:
            images (list): List of images
        """
        test_pipeline = self.config.data.test.pipeline
        test_pipeline = Compose(test_pipeline)
        data = []
        for image in images:
            image_data = dict(img_info=dict(filename=image))
            image_data['seg_fields'] = []
            image_data['img_prefix'] = DOWNLOAD_FOLDER
            image_data['seg_prefix'] = DOWNLOAD_FOLDER
            image_data = test_pipeline(image_data)
            data.append(image_data)
        data = collate(data, samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            data["img_metas"] = [i.data[0] for i in list(data["img_metas"])]

        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        return result

    def show_result_pyplot(
        self,
        img,
        result,
        palette=None,
        fig_size=(15, 10),
        opacity=0.5,
        title="",
        block=True,
        out_file=None,
    ):
        """Visualize the segmentation results on the image.

        Args:
            img (str or np.ndarray): Image filename or loaded image.
            result (list): The segmentation result.
            palette (list[list[int]]] | None): The palette of segmentation
                map. If None is given, random palette will be generated.
                Default: None
            fig_size (tuple): Figure size of the pyplot figure.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
            title (str): The title of pyplot figure.
                Default is ''.
            block (bool): Whether to block the pyplot figure.
                Default is True.
            out_file (str or None): The path to write the image.
                Default: None.
        """
        if hasattr(self.model, "module"):
            model = self.model.module
        img = model.show_result(img, result, palette=palette, show=False, opacity=opacity)
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.title(title)
        plt.tight_layout()
        plt.show(block=block)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
