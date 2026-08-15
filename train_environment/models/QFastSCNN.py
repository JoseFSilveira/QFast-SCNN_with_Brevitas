'''
Based on the modified version of Fast-SCNN in models/FastSCNN.py, changing the tradicional torch.nn (nn) layers to brevitas.nn (qnn) layers.
The following changes to make it compatible with quantization and translation to ONNX and then to FINN:
--> The input and activations are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> The weights are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> Added "qat" mode to the model, which allows for quantization-aware training and export to ONNX and FINN. This mode enables the final upsampling layer to be used.
--> Added "finn" mode to the model, which is used for QONNX export for FINN framework. In this mode, the input preprocessing is done in the model, since FINN expects uint8 inputs.
--> PyramidPooling F.interpolate bilinear with final tensor size as argument is replaced by another F.interpolate with nearest neighbor interpolation with pre-calculated scale factors.
--> The adaptive average pooling layers are replaced by depthwise Conv layers ith kernel weights as 1/kernel_size, which is equivalent to average pooling, but allows for quantization and export to ONNX and FINN.
--> torch.cat() was replaced by a modification of Brevitas qnn.QuantCat to allow for concatenation of QuantTensors, since the QuantTensor class does not have a cat() method anymore (Brevitas Bug).
--> Standart Add operations were replaced by Brevitas qnn.QuantEltwiseAdd to allow for addition of QuantTensors.
--> The last upsampling layer is removed to avoid a large upsampling factor with 'nearest' mode, which can comprimise severly the accuracy of the model.
  obs: The last upsampling layer can be done in external post-processing step, with the output of the model being passed to a CPU or small GPU.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat, Int8BiasPerTensorFloatInternalScaling
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.enum import *

from config import BIT_WIDTH, IM_SIZE, CROP_SIZE


def get_avgpool_callable(channels: int, kernel_size: tuple[int, int] | int, return_quant_tensor=False) -> nn.Module:
    """
    Returns a callable that performs a depthwise convolution with kernel weights as 1/kernel_size, which is equivalent to average pooling, but allows for quantization and export to ONNX and FINN.
    This is used to replace the adaptive average pooling layers in the original model.
    """
    # Define the kernel area for the average pooling operation
    if type(kernel_size) == int:
        kernel_area = kernel_size * kernel_size
    else:
        kernel_area = kernel_size[0] * kernel_size[1]

    # Create the depthwise convolution layer with frozen weights of 1/kernel_area, which is equivalent to average pooling
    pool = qnn.QuantConv2d(in_channels=channels,
                           out_channels=channels,
                           kernel_size=kernel_size,
                           stride=kernel_size,
                           groups=channels,
                           bias=False,
                           bit_width=BIT_WIDTH,
                           weight_quant=Int8WeightPerTensorFloat,
                           return_quant_tensor=return_quant_tensor)

    # Frese the weights of the depthwise convolution layer to 1/kernel_area, which is equivalent to average pooling
    for param in pool.parameters():
        param.requires_grad = False
    nn.init.constant_(pool.weight.data, 1.0/kernel_area)

    return pool

class CustomQuantCat(qnn.QuantCat):
    '''
    A modification of the original qnn.QuantCat replacing QuantTensor.cat() with torch.cat(), since the QuantTensor does not have cat() method anymore (Brevitas Bug).
    '''
    def forward(self,
                tensor_list: list[torch.Tensor] | list[QuantTensor],
                dim: int = 1) -> torch.Tensor | QuantTensor:
        quant_tensor_list = [self.unpack_input(t) for t in tensor_list]
        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler([qt.value for qt in quant_tensor_list])
            self._set_global_is_quant_layer(False)
            return out
        quant_tensor_list = [self.input_quant(qt) for qt in quant_tensor_list]
        # trigger an assert if scale factors and bit widths are None or different
        output = torch.cat(quant_tensor_list, dim=dim)
        quant_output = self.output_quant(output)
        return self.pack_output(quant_output)


class QFastSCNN(nn.Module):

    def __init__(self, num_classes, mode="raw", **kwargs):
        super().__init__()

        # This modified model has modes instead of wrappers to facilitate state dict management when saving and loading the model.
        possible_modes = ["raw", "qat", "finn"]
        assert mode in possible_modes, f"Mode {mode} not supported. Choose one of {possible_modes}."
        self.mode = mode

        self.inp_quant = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)

        # Pre calculating the input normalization to reduce the number of operation in the model. This is done fora a more hardware friendly implementation.
        # The original operation for the forward pass is: x = ((x / 225.0) - self.mean) / self.std)
        # The goal is to only perform x = x * finn_A - finn_B, where finn_A and finn_B are pre-calculated constants.
        # Only used in "finn" mode. FINN expects the input to be uint8, so the normalization is done in the model.
        if self.mode == "finn":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            finn_A = (1.0 / (255.0 * std))
            finn_B = (mean / std)
            self.register_buffer('finn_A', finn_A)
            self.register_buffer('finn_B', finn_B)

    def forward(self, x):
        if self.mode == "qat":
            size = x.size()[2:] # Saves the original size of the input to upsample the output.
        if self.mode == "finn":
            # doing the hardware (finn) friendly version of x = ((x / 225.0) - self.mean) / self.std)
            x = (x * self.finn_A - self.finn_B)
        x = self.inp_quant(x)
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        if self.mode == "qat":
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            qnn.QuantConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, weight_bit_width=BIT_WIDTH, weight_quant=Int8WeightPerTensorFloat, return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(inplace=True, bit_width=BIT_WIDTH, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            qnn.QuantConv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False, weight_bit_width=BIT_WIDTH, weight_quant=Int8WeightPerTensorFloat, return_quant_tensor=True),
            nn.BatchNorm2d(dw_channels),
            qnn.QuantReLU(inplace=True, bit_width=BIT_WIDTH, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantConv2d(dw_channels, out_channels, 1, bias=False, weight_bit_width=BIT_WIDTH, weight_quant=Int8WeightPerTensorFloat, return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(inplace=True, bit_width=BIT_WIDTH, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            qnn.QuantConv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False, weight_bit_width=BIT_WIDTH, weight_quant=Int8WeightPerTensorFloat, return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            qnn.QuantReLU(inplace=True, bit_width=BIT_WIDTH, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            qnn.QuantConv2d(in_channels * t, out_channels, 1, bias=False, weight_bit_width=BIT_WIDTH, weight_quant=Int8WeightPerTensorFloat, return_quant_tensor=True),
            nn.BatchNorm2d(out_channels),
            # quantize the output so it can be used in the skip connection addition
            qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, bit_width=BIT_WIDTH, return_quant_tensor=True)
        )
        
        # Added quantization for the skip connection
        if self.use_shortcut:
            self.add = qnn.QuantEltwiseAdd(bit_width=BIT_WIDTH, return_quant_tensor=True)
        
    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = self.add(x, out)
        return out

class PyramidPooling(nn.Module):
    """
    Pyramid pooling module

    --> The pool sizes were changed from [1x1, 2x2, 3x3, 6x6] to [1x1, 2x2, 4x4, 8x8] for two distint reasons, since for ONNX export, the image size needs to be divisible by the pool size (in this case 32 and 64).
    --> The original model uses adaptive average pooling, which can be replaced by a quantized version of standard Average Pooling with fixes pre defined kernel sizes.
    --> The original model uses F.interpolate, which is being replaced by a QuantConvTranspose2d with frozen weights of torch.ones() to avoid issues when translating the model to ONNX and then to FINN, since onnx runtime outputs an error when a Resize block parameter is empty.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

        # [pool_out_H, pool_out_W] = [in_H, in_W] / kernel_size
        # in_H and in_W are equal to the model input image sizes divided by 32.
        in_H_train = CROP_SIZE[0] // 32
        in_H_test = IM_SIZE[0] // 32
        self.pool_out_H = [1, 2, 4, 8]  # Kernel sizes for the pyramid pooling layers
        self.kernel_size_train = [in_H_train // pool_out for pool_out in self.pool_out_H]  # Kernel sizes for the pyramid pooling layers during training
        self.kernel_size_test = [in_H_test // pool_out for pool_out in self.pool_out_H]  # Kernel sizes for the pyramid pooling layers during testing

        # The original model uses adaptive average pooling, which can be replaced by a standard Average Pooling with pre-defined kernel sizes.
        # Necessary to create new instance for each pool size since the output size is a parameter in the quantized version.

        # For train the input size is [768, 768], so the kernel sizes are [768, 384, 192, 96]
        self.pool1_train = get_avgpool_callable(in_channels, self.kernel_size_train[0], return_quant_tensor=True)
        self.pool2_train = get_avgpool_callable(in_channels, self.kernel_size_train[1], return_quant_tensor=True)
        self.pool3_train = get_avgpool_callable(in_channels, self.kernel_size_train[2], return_quant_tensor=True)
        self.pool4_train = get_avgpool_callable(in_channels, self.kernel_size_train[3], return_quant_tensor=True)

        # For train the input size is [1024, 2048], so the kernel sizes are [1024, 512, 256, 128]
        self.pool1_test = get_avgpool_callable(in_channels, self.kernel_size_test[0], return_quant_tensor=True)
        self.pool2_test = get_avgpool_callable(in_channels, self.kernel_size_test[1], return_quant_tensor=True)
        self.pool3_test = get_avgpool_callable(in_channels, self.kernel_size_test[2], return_quant_tensor=True)
        self.pool4_test = get_avgpool_callable(in_channels, self.kernel_size_test[3], return_quant_tensor=True)
    
        self.concat = CustomQuantCat(bit_width=BIT_WIDTH, return_quant_tensor=True)

    def upsample(self, x, scale_factor):
            return F.interpolate(x, scale_factor=scale_factor, mode='nearest', recompute_scale_factor=False)

    def forward(self, x):

        if self.training:
            feat1 = self.upsample(self.conv1(self.pool1_train(x)), scale_factor=self.kernel_size_train[0])
            feat2 = self.upsample(self.conv2(self.pool2_train(x)), scale_factor=self.kernel_size_train[1])
            feat3 = self.upsample(self.conv3(self.pool3_train(x)), scale_factor=self.kernel_size_train[2])
            feat4 = self.upsample(self.conv4(self.pool4_train(x)), scale_factor=self.kernel_size_train[3])
        else:
            feat1 = self.upsample(self.conv1(self.pool1_test(x)), scale_factor=self.kernel_size_test[0])
            feat2 = self.upsample(self.conv2(self.pool2_test(x)), scale_factor=self.kernel_size_test[1])
            feat3 = self.upsample(self.conv3(self.pool3_test(x)), scale_factor=self.kernel_size_test[2])
            feat4 = self.upsample(self.conv4(self.pool4_test(x)), scale_factor=self.kernel_size_test[3])

        x = self.concat([x, feat1, feat2, feat3, feat4])
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            qnn.QuantConv2d(out_channels, out_channels, 1,
                            weight_bit_width=BIT_WIDTH,
                            weight_quant=Int8WeightPerTensorFloat,
                            return_quant_tensor=True),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            qnn.QuantConv2d(highter_in_channels, out_channels, 1,
                            weight_bit_width=BIT_WIDTH,
                            weight_quant=Int8WeightPerTensorFloat,
                            return_quant_tensor=True),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = qnn.QuantReLU(inplace=True, bit_width=BIT_WIDTH, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)

        # Added quantization for the skip connection
        self.add = qnn.QuantEltwiseAdd(bit_width=BIT_WIDTH, return_quant_tensor=True)

    def upsample(self, x, scale_factor):
                return F.interpolate(x, scale_factor=scale_factor, mode='nearest')                                           

    def forward(self, higher_res_feature, lower_res_feature):

        lower_res_feature = self.upsample(lower_res_feature, scale_factor=self.scale_factor)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)

        out = self.add(higher_res_feature, lower_res_feature)
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            qnn.QuantConv2d(dw_channels, num_classes, 1,
                            weight_bit_width=BIT_WIDTH,
                            weight_quant=Int8WeightPerTensorFloat,
                            return_quant_tensor=False)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x