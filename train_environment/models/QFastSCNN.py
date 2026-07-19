'''
Based on the modified version of Fast-SCNN in models/FastSCNN.py, changing the tradicional torch.nn (nn) layers to brevitas.nn (qnn) layers.
The following changes to make it compatible with quantization and translation to ONNX and then to FINN:
--> The input and activations are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> The weights are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> Added "qat" mode to the model, which allows for quantization-aware training and export to ONNX and FINN. This mode enables the final upsampling layer to be used.
--> Added "finn" mode to the model, which is used for QONNX export for FINN framework. In this mode, the input preprocessing is done in the model, since FINN expects uint8 inputs.
--> F.interpolate is replaced by depthwise qnn.QuantConvTranspose2d with frozen weights of torch.ones().
--> The adaptive average pooling layers are replaced by qnn.TruncAvgPool2d with pre-calculated kernel sizes.
--> Pooling operations are quantized to avoid issues when translating the model to ONNX and then to FINN.
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

from config import BIT_WIDTH, IM_SIZE


def get_upsample_callable(channels: int, scale_factor: int | tuple[int, int]) -> qnn.QuantConvTranspose2d:
    """
    Returns a callable that performs upsampling using a quantized depthwise transposed convolution with frozen weights of torch.ones(). This is used to replace F.interpolate in the model, which is not supported by FINN.
    """
    return qnn.QuantConvTranspose2d(channels, channels, scale_factor,
                                    stride=scale_factor,
                                    padding=0, 
                                    groups=channels,
                                    bias=False,
                                    weight_bit_width=BIT_WIDTH,
                                    weight_quant=Int8WeightPerTensorFloat,
                                    return_quant_tensor=True)


class ParameterizedActQuant(Int8ActPerTensorFloat):
    '''
    Custom parameterized activation quantizer that restricts the scaling implementation type to be a parameter, which is necessary for the quantization-aware training and export to ONNX and FINN.
    '''
    scaling_impl_type = ScalingImplType.PARAMETER
    max_val = 6.0
    min_val = -6.0


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
            nn.BatchNorm2d(out_channels)
        )
        
        # Added quantization for the skip connection
        self.shared_quant = qnn.QuantIdentity(act_quant=ParameterizedActQuant,
                                               bit_width=BIT_WIDTH,
                                               return_quant_tensor=True)
        
        self.out_quant = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                               bit_width=BIT_WIDTH,
                                               return_quant_tensor=True)
        
    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            x = self.shared_quant(x)
            out = self.shared_quant(out)
            out = x + out
        return self.out_quant(out)

class PyramidPooling(nn.Module):
    """
    Pyramid pooling module
    --> The pool sizes were changed from [1, 2, 3, 6] to [1, 2, 4, 8] since for onnx export the img size needs to be divisible by the pool size (in this case 32 and 64).
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

        self.pool_output_sizes = [1, 2, 4, 8]  # Pool sizes for the pyramid pooling layers

        # Defining the output size for the PiramidPooling class
        tensor_size = tuple(im_size // 32 for im_size in IM_SIZE) # The original model uses a fixed input size of 1024x2048, which is downsampled by a factor of 32 before the PyramidPooling.

        # Defining the kernel sizes for the pooling layers, which are also the same as the scale factor for upsampling layers.
        # The original model uses adaptive average pooling, which can be replaced by a quantized version of it.
        self.kernel_sizes = []
        for pool_out_size in self.pool_output_sizes:
            self.kernel_sizes.append(tuple(size // pool_out_size for size in tensor_size))

        # Added quantization for the pooling layers. The original model uses adaptive average pooling, which can be replaced by a quantized version of it.
        # Necessary to create new instance for each pool size since the output size is a parameter in the quantized version.
        self.pool1 = qnn.TruncAvgPool2d(self.kernel_sizes[0], return_quant_tensor=True)
        self.pool2 = qnn.TruncAvgPool2d(self.kernel_sizes[1], return_quant_tensor=True)
        self.pool4 = qnn.TruncAvgPool2d(self.kernel_sizes[2], return_quant_tensor=True)
        self.pool8 = qnn.TruncAvgPool2d(self.kernel_sizes[3], return_quant_tensor=True)

        ## -- BEGIN OF QUANTIZED UPSAMPLE DEFINITION -- ##
        '''
        Added quantization for the upsampled features. The original model uses F.interpolate, which is being replaced by a QuantConvTranspose2d with frozen weights of torch.ones()
        This modification is to avoid issues when translating the model to ONNX and then to FINN, since onnx runtime outputs an error when a Resize block parameter is empty.
        '''

        # Defining the upsample layers for each pooling layer. The original model uses F.interpolate, which is being replaced by a QuantConvTranspose2d with frozen weights of torch.ones().
        # The scale factors are the same as the kernel sizes for the pooling layers, since the output size of the pooling layers is the same as the input size of the upsample layers.
        self.upsample1 = get_upsample_callable(inter_channels, self.kernel_sizes[0])
        self.upsample2 = get_upsample_callable(inter_channels, self.kernel_sizes[1])
        self.upsample4 = get_upsample_callable(inter_channels, self.kernel_sizes[2])
        self.upsample8 = get_upsample_callable(inter_channels, self.kernel_sizes[3])

        # Initialize the weights to 1.0 and freeze them to mimic the behavior of F.interpolate with mode='nearest'.
        nn.init.constant_(self.upsample1.weight, 1.0)
        for param in self.upsample1.parameters():
            param.requires_grad = False
        nn.init.constant_(self.upsample2.weight, 1.0)
        for param in self.upsample2.parameters():
            param.requires_grad = False
        nn.init.constant_(self.upsample4.weight, 1.0)
        for param in self.upsample4.parameters():
            param.requires_grad = False
        nn.init.constant_(self.upsample8.weight, 1.0)
        for param in self.upsample8.parameters():
            param.requires_grad = False

        ## -- END OF QUANTIZED UPSAMPLE DEFINITION -- ##

        #self.shared_act = qnn.QuantIdentity(act_quant=ParameterizedActQuant,
        #                                    bit_width=BIT_WIDTH,
        #                                    return_quant_tensor=True)

        self.concat = qnn.QuantCat(return_quant_tensor=True)

        self.quant_out = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                           bit_width=BIT_WIDTH,
                                           return_quant_tensor=True)

    def forward(self, x):

        feat1 = self.shared_act(self.upsample1(self.conv1(self.pool1(x))))
        feat2 = self.shared_act(self.upsample2(self.conv2(self.pool2(x))))
        feat3 = self.shared_act(self.upsample4(self.conv3(self.pool4(x))))
        feat4 = self.shared_act(self.upsample8(self.conv4(self.pool8(x))))

        #x = self.shared_act(x)
        #x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.concat([x, feat1, feat2, feat3, feat4])
        #x = self.quant_out(x)
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

        # Added quantization for the upsampled features. The original model uses F.interpolate, which is being replaced by a QuantConvTranspose2d with frozen weights of torch.ones()
        # This modification is to avoid issues when translating the model to ONNX and then to FINN, since onnx runtime outputs an error when a Resize block parameter is empty.
        self.upsample = get_upsample_callable(lower_in_channels, scale_factor)
        nn.init.constant_(self.upsample.weight, 1.0) # Initialize the weights to 1.0 to mimic the behavior of F.interpolate with mode='nearest'.
        for param in self.upsample.parameters():
            param.requires_grad = False # freeze the weights to avoid training them, since they are not learnable parameters.

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
        self.shared_quant = qnn.QuantIdentity(act_quant=ParameterizedActQuant,
                                               bit_width=BIT_WIDTH,
                                               return_quant_tensor=True)
                                           

    def forward(self, higher_res_feature, lower_res_feature):

        lower_res_feature = self.upsample(lower_res_feature)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)

        # Added quantization for the features before the fusion
        lower_res_feature = self.shared_quant(lower_res_feature)
        higher_res_feature = self.shared_quant(higher_res_feature)

        out = higher_res_feature + lower_res_feature
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