'''
Based on the modified version of Fast-SCNN in models/FastSCNN.py, changing the tradicional torch.nn (nn) layers to brevitas.nn (qnn) layers.
The following changes to make it compatible with quantization and translation to ONNX and then to FINN:
--> The input and activations are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> The weights are quantized to 8 bits using per-tensor quantization with a floating-point scale.
--> Added "qat" mode to the model, which allows for quantization-aware training and export to ONNX and FINN. This mode enables the final upsampling layer to be used.
--> Added "finn" mode to the model, which is used for QONNX export for FINN framework. In this mode, the input preprocessing is done in the model, since FINN expects uint8 inputs.
--> F.interpolate is replaced by depthwise qnn.QuantConvTranspose2d with frozen weights of torch.ones().
--> The adaptive average pooling layers are replaced by nn.AvgPool2d with pre-calculated kernel sizes.
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
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3], mode=mode)
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
    --> The pool sizes were changed from [1, 2, 3, 6] to [1, 2, 4, 8] since for onnx export the img size needs to be divisible by the pool size (in this case 32 and 64).
    --> The original model uses adaptive average pooling, which can be replaced by a quantized version of standard Average Pooling with fixes pre defined kernel sizes.
    --> The original model uses F.interpolate, which is being replaced by a QuantConvTranspose2d with frozen weights of torch.ones() to avoid issues when translating the model to ONNX and then to FINN, since onnx runtime outputs an error when a Resize block parameter is empty.
    """

    def __init__(self, in_channels, out_channels, mode='raw', **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

        self.mode = mode
        self.pool_output_sizes = [1, 2, 4, 8]  # Pool sizes for the pyramid pooling layers

        ''' BEGIN OF TRAIN PARAMETERS DEFINITION '''

        # Defining the output size for the PiramidPooling class
        self.tensor_size_train = tuple(im_size // 32 for im_size in CROP_SIZE) # During Training the original model uses a fixed input size of 768x768, which is downsampled by a factor of 32 before the PyramidPooling.

        # Defining the kernel sizes for the pooling layers, which are also the same as the scale factor for upsampling layers.
        # The original model uses adaptive average pooling, which can be replaced by a quantized version of it.
        self.kernel_sizes_train = []
        for pool_out_size in self.pool_output_sizes:
            self.kernel_sizes_train.append(tuple(size // pool_out_size for size in self.tensor_size_train))

        # The original model uses adaptive average pooling, which can be replaced by a standard Average Pooling with pre-defined kernel sizes.
        # Necessary to create new instance for each pool size since the output size is a parameter in the quantized version.
        self.pool1_train = nn.AvgPool2d(self.kernel_sizes_train[0])
        self.pool2_train = nn.AvgPool2d(self.kernel_sizes_train[1])
        self.pool4_train = nn.AvgPool2d(self.kernel_sizes_train[2])
        self.pool8_train = nn.AvgPool2d(self.kernel_sizes_train[3])

        # Added quantization for the upsampled features. The original model uses F.interpolate, which is being replaced by a QuantConvTranspose2d with frozen weights of torch.ones()
        # This modification is to avoid issues when translating the model to ONNX and then to FINN, since onnx runtime outputs an error when a Resize block parameter is empty.
        
        # Defining the upsample layers for each pooling layer. The original model uses F.interpolate, which is being replaced by a QuantConvTranspose2d with frozen weights of torch.ones().
        # The scale factors are the same as the kernel sizes for the pooling layers, since the output size of the pooling layers is the same as the input size of the upsample layers.
        self.upsample1_train = get_upsample_callable(inter_channels, self.kernel_sizes_train[0])
        self.upsample2_train = get_upsample_callable(inter_channels, self.kernel_sizes_train[1])
        self.upsample4_train = get_upsample_callable(inter_channels, self.kernel_sizes_train[2])
        self.upsample8_train = get_upsample_callable(inter_channels, self.kernel_sizes_train[3])

        # Initialize the weights to 1.0 and freeze them to mimic the behavior of F.interpolate with mode='nearest'.
        for op in [self.upsample1_train, self.upsample2_train, self.upsample4_train, self.upsample8_train]:
            nn.init.constant_(op.weight, 1.0)
            for param in op.parameters():
                param.requires_grad = False

        ''' END OF TRAIN PARAMETERS DEFINITION '''
        
        ''' BEGIN OF EVAL PARAMETERS DEFINITION '''
        
        self.tensor_size_eval = tuple(im_size // 32 for im_size in IM_SIZE) # During Eval the original model uses a fixed input size of 1024x2048, which is downsampled by a factor of 32 before the PyramidPooling.

        self.kernel_sizes_eval = []
        for pool_out_size in self.pool_output_sizes:
            self.kernel_sizes_eval.append(tuple(size // pool_out_size for size in self.tensor_size_eval))

        self.pool1_eval = nn.AvgPool2d(self.kernel_sizes_eval[0])
        self.pool2_eval = nn.AvgPool2d(self.kernel_sizes_eval[1])
        self.pool4_eval = nn.AvgPool2d(self.kernel_sizes_eval[2])
        self.pool8_eval = nn.AvgPool2d(self.kernel_sizes_eval[3])

        self.upsample1_eval = get_upsample_callable(inter_channels, self.kernel_sizes_eval[0])
        self.upsample2_eval = get_upsample_callable(inter_channels, self.kernel_sizes_eval[1])
        self.upsample4_eval = get_upsample_callable(inter_channels, self.kernel_sizes_eval[2])
        self.upsample8_eval = get_upsample_callable(inter_channels, self.kernel_sizes_eval[3])

        for op in [self.upsample1_eval, self.upsample2_eval, self.upsample4_eval, self.upsample8_eval]:
            nn.init.constant_(op.weight, 1.0)
            for param in op.parameters():
                param.requires_grad = False

        ''' END OF EVAL PARAMETERS DEFINITION '''
    
        self.concat = CustomQuantCat(bit_width=BIT_WIDTH, return_quant_tensor=True)

        self.quant_tensor = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                           bit_width=BIT_WIDTH,
                                           return_quant_tensor=True)

    def forward(self, x):

        if self.training:
            # Use CROP_SIZE for training [768x768]
            feat1 = self.upsample1_train(self.conv1(self.quant_tensor(self.pool1_train(x))))
            feat2 = self.upsample2_train(self.conv2(self.quant_tensor(self.pool2_train(x))))
            feat3 = self.upsample4_train(self.conv3(self.quant_tensor(self.pool4_train(x))))
            feat4 = self.upsample8_train(self.conv4(self.quant_tensor(self.pool8_train(x))))
        else:
            # Use IM_SIZE for evaluation [1024x2048]
            feat1 = self.upsample1_eval(self.conv1(self.quant_tensor(self.pool1_eval(x))))
            feat2 = self.upsample2_eval(self.conv2(self.quant_tensor(self.pool2_eval(x))))
            feat3 = self.upsample4_eval(self.conv3(self.quant_tensor(self.pool4_eval(x))))
            feat4 = self.upsample8_eval(self.conv4(self.quant_tensor(self.pool8_eval(x))))

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
                 out_channels=128, t=6, num_blocks=(3, 3, 3), mode='raw', **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels, mode=mode)

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
        self.add = qnn.QuantEltwiseAdd(bit_width=BIT_WIDTH, return_quant_tensor=True)
                                           

    def forward(self, higher_res_feature, lower_res_feature):

        lower_res_feature = self.upsample(lower_res_feature)
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