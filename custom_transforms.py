import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision import tv_tensors  # <--- Importação necessária pára Augmentations
from config import IM_HEIGHT, IM_WIDTH, CROP_SIZE, IGNORE_INDEX

# -- Funcoes personalizadas para transformacoes -- #

# -- 1. remove canal extra desnecessário na segmentação
mask_squeeze = lambda x: x.squeeze(dim=0)

# -- 2. transforma os ids originais para ids de treino
class IdToTrainIdTransform:

    def __init__(self, conv_dict: dict):
        self.conv_dict = conv_dict

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        new_mask = mask # cria uma copia da mascara original para ser modificada
        # Troca os valores da mascara original a partir do dicionario
        for lable, new_lable in list(self.conv_dict.items()):
            # Caso o pixel esteja com a mascara da key, troca para a mascara do value, caso contrario mantem a mascara original
            new_mask = torch.where(mask == lable, new_lable, new_mask)
        return new_mask
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Transforms:

    def __init__(self, conv_size: tuple[int], lable_conversion: dict):

        # transformando dados em tensores e aplicando data augmentation
        # O dataset original possui tamanho 1024x2048

        # -- DATA TRANSFORMS -- #
        self.train_transform = v2.Compose([
            #v2.Resize(size=conv_size, interpolation=InterpolationMode.BILINEAR), # redimensiona imagem
            v2.PILToTensor(), # converte imagem PIL para tensor
            v2.ToDtype(torch.uint8) # apenas converte para inteiro sem normalizacao. Isso eh feito pois o algumas funcoes do DataAugmentation exigem que a imagem seja do tipo uint8
            # OBS: o restante da normalizacao (dividir por 255 e normalizar com media e desvio padrao) eh feito APOS a data augmentation
        ])

        self.val_transform = v2.Compose([
            #v2.Resize(size=conv_size, interpolation=InterpolationMode.BILINEAR), # redimensiona imagem para 256x512
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normaliza com media e desvio padrao do ImageNet
        ])

        # -- TARGET TRANFROMS -- #
        self.target_transform = v2.Compose([
            #v2.Resize(size=conv_size, interpolation=InterpolationMode.NEAREST_EXACT), # redimensiona imagem para 256x512. Nearest Neighbor para nao criar novos valores
            v2.PILToTensor(), # converte segmentação PIL para tensor
            IdToTrainIdTransform(lable_conversion), # converte ids originais para ids de treino
            v2.Lambda(mask_squeeze), # remove canal extra desnecessário na segmentação
            v2.ToDtype(torch.uint8) # apenas converte para inteiro sem normalizacao
        ])

        # -- DATA AUGMENTATION -- #
        self.data_augmentation = v2.Compose([

            # Transformações Geométricas
            v2.RandomHorizontalFlip(p=0.5),
            v2.ScaleJitter(target_size=(IM_HEIGHT, IM_WIDTH), scale_range=(0.5, 2.0), antialias=True), # Redimensiona aleatoriamente entre 50% e 200% do tamanho base
            v2.RandomRotation(degrees=2, interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill={tv_tensors.Image: (0,0,0), tv_tensors.Mask: IGNORE_INDEX}),
            v2.RandomCrop(size=CROP_SIZE, pad_if_needed=True,
                          fill={tv_tensors.Image: (0,0,0), tv_tensors.Mask: IGNORE_INDEX}), # Corta aleatoriamente uma região de 768x768, preenchendo com preto ou IGNORE_INDEX se necessário
            
            # Transformações Fotométricas (O v2 aplica AUTOMATICAMENTE só na Imagem)
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            
            # Opcional: Borrão para simular foco ruim
            v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ])

        # -- POST DATA AUGMENTATION TRANSFORMS -- #
        self.post_data_augmentation = v2.Compose([
            v2.ToDtype(torch.float32, scale=True), # Converte para float e normaliza para [0,1]
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normaliza com media e desvio padrao do ImageNet
        ])