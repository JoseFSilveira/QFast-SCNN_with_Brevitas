import torch
from torchvision import datasets
from torchvision.transforms import v2
from torchvision import tv_tensors  # <--- Importação necessária paara Augmentations
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm.asyncio import tqdm
from config import NUM_CLASSES, IGNORE_INDEX


class CityscapesLables:
    def __init__(self):

        # Definindo o numero de classes treinaveis (19 classes + 1 classe de ignorar)
        self.num_classes = NUM_CLASSES
        self.ignore_index = IGNORE_INDEX

        # Criando listas dos nomes e das cores das classes treinaveis
        id_names = {}
        color_list = []
        lable_conversion = {}
        for c in datasets.Cityscapes.classes:

            # Adicionando valores ao dicionario de conversao de ids
            lable_conversion[c.id] = c.train_id if c.train_id != -1 else self.ignore_index # A classe 'ignore' tem train_id -1, entao atribui o valor IGNORE_INDEX para ela
            # Adicionando valores as listas de nomes e cores
            if c.train_id != -1 and c.train_id != 255:
                id_names[c.train_id] = c.name
                color_list.append(c.color)

        # Variavel para dicionario de nomes
        id_names.update({self.ignore_index: 'ignore'}) # Adiciona a classe 'ignore' com train_id 255
        self.id_names = id_names

        # Variavel para lista de cores
        train_colors_list = color_list
        train_colors_list.append((0,0,0)) # Adiciona a cor preta para a classe 'ignore'. O cmap funcionara pois 255 constara como overflow e sera mapeado para a ultima cor da lista, que eh a preta
        self.train_colors_list = train_colors_list
        self.train_color_map = ['#%02x%02x%02x' % color for color in self.train_colors_list] # Criando o color map para imprimir imagens segmentadas

        # Criando o mapeamento para labels de treino
        self.lable_conversion = lable_conversion


    def get_cmap(self, name='CityscapesTrainColorMap'):
        # Criando o color map para imprimir imagens segmentadas
        return mcolors.LinearSegmentedColormap.from_list(name, self.train_color_map, N=len(self.train_color_map))
    

    def get_histogram(self, dataloader: datasets.Cityscapes, print_histogram=False, save_path=None, device='cpu'):

        '''
        Cria histograma de frequencia das classes no dataset, para poder usar class_weights
        Eh esperado que o dataset ja esteja com as mascaras originais convertidas para 20 Classes (19 + ignore) usando o IdToTrainIdTransform
        '''

        # Verificar se o dataset tem mascaras convertidas para 20 classes, caso contrario o histograma sera incorreto
        _, mask = next(iter(dataloader))
        if mask.max() != 255 or mask.min() < 0:
            raise ValueError(f"Dataset com mascaras nao convertidas para {self.num_classes+1} classes. O histograma sera incorreto. Use o IdToTrainIdTransform para converter as mascaras antes de criar o dataset.")

        # Criar histograma de frequencia das classes no dataset
        class_count = torch.zeros(self.num_classes+1, dtype=torch.long, device=device) # Inicializa o contador de pixels para cada classe como um tensor de zeros
        for _, mask in tqdm(dataloader, desc="Calculando histograma"):
            mask = mask.to(device) # Aloca as mascaras do batch no dispositivo
            for c in range(self.num_classes+1):
                # Conta o numero de pixels da classe c no batch e adiciona ao contador total. Para a classe 'ignore' (train_id 255), conta os pixels com valor 255
                class_count[c] += (mask == c).sum() if c != self.num_classes else (mask == self.ignore_index).sum()

        # Cria o grafico se print_histogram for True ou save_path for fornecido
        if print_histogram or save_path is not None:
            plt.figure(figsize=(10, 5))
            plt.bar(self.id_names.values(), class_count.cpu().numpy(), color=self.train_color_map, log=True)
            plt.xlabel('classes')
            plt.xticks(rotation=90)
            plt.ylabel('numero de pixels (log scale)')
            plt.title('Histograma de frequencia das classes no dataset')
            plt.grid(True)

            # Salva a imagem caso a variavel save_path for nao nula
            if save_path is not None:
                img_file = Path(save_path) / "dataset_histogram.png"
                plt.savefig(img_file, bbox_inches='tight')

            # Imprime o histograma se print_histogram for True
            if print_histogram:
                plt.show()
            else:
                plt.close() # Fecha a figura para liberar memoria se nao for mostrar

        return class_count
    

    def get_weights(self, dataloader: datasets.Cityscapes, method: str='enet', print_histogram: bool=False, save_path: str=None, device: str='cpu') -> torch.Tensor:
        '''
        Cria pesos para as classes a partir do histograma de frequencia das classes no dataset
        Eh esperado que o dataset ja esteja com as mascaras originais convertidas para 20 Classes (19 + ignore) usando o IdToTrainIdTransform
        O metodo 'enet' eh baseado no paper "ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation" (https://arxiv.org/abs/1606.02147)
        O metodo 'median_freq' eh baseado no paper "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation" (https://arxiv.org/abs/1511.00561)
        '''

        class_count = self.get_histogram(dataloader, print_histogram, save_path, device=device)

        if method == 'enet':
            total_count = class_count.sum()
            class_weights = 1 / torch.log(1.02 + class_count / total_count) # A constante 1.02 eh usada para evitar divisao por zero e suavizar os pesos

        elif method == 'median_freq':
            median_freq = torch.median(class_count[class_count > 0]) # Calcula a mediana apenas das classes presentes no dataset
            class_weights = median_freq / class_count
            class_weights[class_count == 0] = 0 # Define peso zero para classes ausentes

        else:
            raise ValueError(f"Metodo desconhecido: {method}. Use 'enet' ou 'median_freq'.")
        
        # Define peso zero para a classe 'ignore', que tem train_id 19 (ultima classe treinavel)
        class_weights[-1] = 0
        
        # Salva os pesos em um arquivo txt, caso save_path seja fornecido
        if save_path is not None:
            
            # Cria um dicionario com os nomes das classes e seus pesos correspondentes e o salva como json, para facilitar a leitura dos pesos posteriormente
            class_dict = {self.id_names[i]: class_weights[i].item() for i in range(self.num_classes)}

            # Salva o json em um arquivo
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            weights_file = save_path / Path(f"class_weights_{method}.json")
            with open(weights_file, 'w') as f:
                json.dump(class_dict, f, indent=4)

            print(f"Pesos das classes salvos no arquivo {weights_file}")
            
        return class_weights


# Criar Classe para dateset modificado
class AugmentedCityscapes(datasets.Cityscapes):
    def __init__(self, *args, data_augmentation: v2.Compose=None, post_data_augmentation: v2.Compose=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_augmentation = data_augmentation
        self.post_data_augmentation = post_data_augmentation

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)

        # Aplicar data augmentation se estiver definido
        if self.data_augmentation is not None:

            # tv_tensors informama ao v2 que o 'image' é uma Imagem e 'target' é uma Mascara.
            # Isso garante que rotacoes/flips sejam aplicados em ambos,
            # e que ajustes de cor (se houver) sejam aplicados APENAS na imagem.
            image, target = self.data_augmentation(tv_tensors.Image(image), tv_tensors.Mask(target))
            
        if self.post_data_augmentation is not None:
            image = self.post_data_augmentation(image)

        return image, target