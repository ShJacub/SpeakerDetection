from torch.utils.data import Dataset

import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import torch


class VideoDecoder(Dataset):
    '''
    VideoProcessingFramework: https://github.com/NVIDIA/VideoProcessingFramework
    '''
    def __init__(self, video_path, gpu_id=0, fp32=True):
        self.video_path = video_path
        self.gpu_id = gpu_id
        self.nvdecoder = None
        self.from_nv12_to_yuv = None
        self.from_yuv420_to_rgb = None
        self.from_rgb_to_pln = None
        self.cc_ctx = None
        self.frame_idx = 0
        self.fp32 = fp32

    def init(self):
        # Init HW decoder, convertor, resizer and color space context:
        self.nvdecoder = nvc.PyNvDecoder(input=self.video_path, gpu_id=self.gpu_id)
        self.height, self.width = self.nvdecoder.Height(), self.nvdecoder.Width()

        self.from_nv12_to_yuv = nvc.PySurfaceConverter(
            self.width, self.height, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, self.gpu_id
        )
        self.from_yuv420_to_rgb = nvc.PySurfaceConverter(
            self.width, self.height, nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, self.gpu_id
        )
        self.from_rgb_to_pln = nvc.PySurfaceConverter(
            self.width, self.height, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, self.gpu_id
        )
        self.cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG
        )

        return self

    def __len__(self):
        return self.nvdecoder.Numframes()

    def get_fps(self):
        return self.nvdecoder.Framerate()
    
    def get_img_size(self):
        return self.width, self.height

    def __getitem__(self, idx):
        if idx >= len(self):
            raise RuntimeError("idx is bigger than length of video")

        # Decode 1 compressed video frame to CUDA memory:
        nv12_surface = self.nvdecoder.DecodeSingleSurface()
        if nv12_surface.Empty():
            print("Can not decode frame")
            return None

        # Convert from NV12 to YUV420
        # This extra step is required because not all NV12 -> RGB conversions
        # implemented in NPP support all color spaces and ranges:
        yuv420 = self.from_nv12_to_yuv.Execute(nv12_surface, self.cc_ctx)
        if yuv420.Empty():
            print("Can not convert nv12 -> yuv420")
            return None

        # Convert from YUV420 to interleaved RGB:
        rgb24 = self.from_yuv420_to_rgb.Execute(yuv420, self.cc_ctx)
        if rgb24.Empty():
            print("Can not convert yuv420 -> rgb")
            return None

        # Convert from RGB to planar RGB:
        rgb24_planar = self.from_rgb_to_pln.Execute(rgb24, self.cc_ctx)
        if rgb24_planar.Empty():
            print("Can not convert rgb -> rgb planar")
            return None


        # Without this, the GPU memory will be overwritten every time, and the tensor will be corrupted:
        rgb24_planar = rgb24_planar.Clone()


        # Export to PyTorch tensor:
        surf_plane = rgb24_planar.PlanePtr()
        img_tensor = pnvc.makefromDevicePtrUint8(
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )
        if img_tensor is None:
            raise RuntimeError("Can not export to tensor.")

        # Form tensor:
        img_tensor.resize_(3, self.height, self.width)
        img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor if self.fp32 else torch.cuda.DoubleTensor)

        # Count read frames:
        self.frame_idx += 1

        return {'img': img_tensor}