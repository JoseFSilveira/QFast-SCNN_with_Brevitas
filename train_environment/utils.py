import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from pathlib import Path
# Mais informacoes em https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html#torchmetrics.classification.MulticlassJaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
import warnings
import netron
import time
from IPython.display import IFrame

import custom_cityscapes as ccs

def plot_leaning_rate_evolution(learning_rates: list[float]) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot(learning_rates)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    x_ticks_interval = len(learning_rates) // 10
    plt.xticks(range(1, len(learning_rates) + 1, x_ticks_interval))
    plt.grid(True, which='major')
    plt.title("Evolucao do Learning Rate ao longo das Epochs")
    plt.show()


def print_results(model_results: dict, metrics: dict) -> None:

    if model_results is not None:
        
        # Gerar Figuras
        plt.figure(figsize=(5,5))
        fig, ax = plt.subplots(nrows=1, ncols=len(metrics)+1, figsize=(15,5))

        epochs = range(1, len(model_results['train_loss'])+1)

        # Para evitar poluicao visual definde xticks para 10% dos epochs, de modo a ter 10 referencias visuais ao lonfo da curva, independente da quantidade de epochs treinadas
        x_ticks_interval = len(model_results['train_loss']) // 10
        x_ticks = range(1, len(model_results['train_loss'])+1, x_ticks_interval)
        
        # Plotar as curvas de perda de treino e validacao no primeiro subplot
        ax[0].scatter(epochs, model_results['train_loss'], label='Train Loss', c='blue')
        ax[0].scatter(epochs, model_results['val_loss'], label='Val Loss', c='orange')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_xticks(x_ticks)
        ax[0].grid(True)
        ax[0].legend()
        ax[0].set_title('Loss Curves')

        # Plotar as curvas de cada metrica de validacao nos subplots seguintes
        for i, (metric_name, metric_fn) in enumerate(metrics.items()):
            metric_fn.plot(model_results[f'val_{metric_name}'], ax=ax[i+1])
            ax[i+1].set_xlabel('Epoch')
            ax[i+1].set_ylabel(f'val_{metric_name}')
            ax[i+1].set_xticks(x_ticks) # Evita Poluicao Visual
            ax[i+1].grid(True, which='major') # Evita Poluicao Visual
            if metric_name == 'IoU':
                ax[i+1].legend(labels=ccs.CityscapesLables().id_names.values(), loc='lower left', ncols=2, fontsize="xx-small") # Adiciona legenda com o nome de cada classe, para facilitar a interpretacao das curvas de IoU de cada classe
            ax[i+1].set_title(f'{metric_name} Curves')

        plt.show()

    else:
        print("Resultados do modelo nao encontrados.")


# Imprimindo imagens e segmentações do dataset
def img_show(imgs: list[torch.Tensor], smnts1: list[torch.Tensor], smnts2: list[torch.Tensor]=None, n: int=5, col_names: list = None, cmap: mcolors.LinearSegmentedColormap = None) -> None:

    '''
    Essa funcao imprime as imagens e segmentacoes do dataset,
    caso smnts2 seja fornecido, entao as segmentacoes de smnts1 e smnts2 sao impressas lado a lado,
    caso contrario, apenas as segmentacoes de smnts1 sao impressas
    '''

    # Definindo Variavel auxiliar paraa definir se os titulos das colunas devem er mostrados ou nao, dependendo se col_names foi fornecido e se a quantidade de nomes informados estiver correta
    if col_names is not None:
        if smnts2 is not None and len(col_names) == 3:
            show_col_names = True
        elif smnts2 is None and len(col_names) == 2:
            show_col_names = True
        else:
            show_col_names = False
            print("Quantidade de nomes de colunas informados nao corresponde a quantidade de colunas a serem mostradas. Os titulos das colunas nao serao mostrados.")

    # Definindo o numero de colunas e o titulo de cada coluna, dependendo se smnts2 foi fornecido ou nao
    if smnts2 is not None:
        fig_dim = 15
        num_cols = 3
    else:
        fig_dim = 10
        num_cols = 2

    # Criando a figura e os eixos para imprimir as imagens e segmentacoes
    fig, axes = plt.subplots(nrows=n, ncols=num_cols, figsize=(fig_dim, fig_dim))

    # Colocando Titulo para cada Coluna caso col_names tenha sido fornecido corretamente
    if show_col_names:
        for ax, col in zip(axes[0], col_names):
            ax.set_title(col)

    # Parametros paara desnormalizar imagens
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1) # Reshape para (3,1,1) para que a operacao de desnormalizacao seja aplicada corretamente em cada canal da imagem
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) # Reshape para (3,1,1)

    for i in range(n):
        
        img_to_show = imgs[i] * std + mean # Desnormaliza a imagem para que as cores sejam mostradas corretamente
        #img_to_show = torch.clamp(img_to_show, 0, 1) # Garante que qualquer residuo matematico fique estritamente entre 0 e 1

        axes[i, 0].imshow(img_to_show.permute(1,2,0)) # permute para mudar a ordem dos canais e converter um tensor para imagem
        axes[i, 0].axis('off')
        axes[i, 1].imshow(smnts1[i], cmap=cmap, vmin=0, vmax=19) # vmin e vmax para garantir que a segmentacao seja mostrada com as mesmas cores, independente da quantidade de classes presentes em cada segmentacao
        axes[i, 1].axis('off')
        if smnts2 is not None:
            axes[i, 2].imshow(smnts2[i], cmap=cmap, vmin=0, vmax=19)
            axes[i, 2].axis('off')
    plt.show()


def predict_mask(model: torch.nn.Module, img: torch.Tensor, device: torch.device='cpu') -> torch.Tensor:
    smnt_logits = model(img.unsqueeze(dim=0).to(device))
    smnt_mask = torch.softmax(smnt_logits, dim=1).argmax(dim=1).squeeze(dim=0).cpu()
    return smnt_mask


def test_model(model: torch.nn.Module, dataset, n:int = 5, device: torch.device='cpu', cmap: mcolors.LinearSegmentedColormap = None) -> None:
    img_list = []
    smnt_list = []

    model.eval()

    with torch.no_grad():
        # Gera as listas de imagens e mascaras e chapa a funcao de imprimir
        for i in random.sample(range(len(dataset)), k=n):
            img = dataset[i][0]
            smnt_mask = predict_mask(model, img, device)
            img_list.append(img)
            smnt_list.append(smnt_mask)
    
    img_show(imgs=img_list, smnts1=smnt_list, n=n, col_names=['Imagem Original', 'Predicao'], cmap=cmap)


def dataset_show(dataset, n:int = 5, predict_masks: bool=False, model: torch.nn.Module=None, device: torch.device='cpu', cmap: mcolors.LinearSegmentedColormap = None) -> None:

    img_list = []
    smnt_list = []
    for i in random.sample(range(len(dataset)), k=n):
        img, smnt = dataset[i]
        img_list.append(img)
        smnt_list.append(smnt)

    # Realiza as predicoes do modelo, caso seja fornecido um
    if predict_masks and model is not None:

        model.eval()
        pred_smnt_list = []
        with torch.no_grad():
            # Gerando as mascaras preditas pelo modelo para cada imagem do dataset
            for img in img_list:
                pred_smnt = predict_mask(model, img, device)
                pred_smnt_list.append(pred_smnt)

        col_names = ['Imagem Original', 'Ground Truth', 'Predicao']
    
    # Caso contrario apenas aloca None para a lista de predicoes, para que a funcao de imprimir saiba que nao deve imprimir as mascaras preditas
    else:
        pred_smnt_list = None
        col_names = ['Imagem Original', 'Ground Truth']

    img_show(imgs=img_list, smnts1=smnt_list, smnts2=pred_smnt_list,
             n=n,col_names=col_names, cmap=cmap)
    
def load_state_dict(model: torch.nn.Module, path: str, strict: bool = True, ignore_key_name: list=None) -> torch.nn.Module | tuple[torch.nn.Module, dict]:

    # Carregando apenas os parametros (state_dict()), pois isso flexibiliza o modelo e evita erros de incompatibilidade com parametros e caminhos do modelo original
    # OBS: torch.load() carrega o modelo inteiro, nao apenas os parametros
    path = Path(path)
    model_name = path.stem # Extrai o nome do modelo a partir do caminho dado

    if path.is_file():
        print(f"Carregando modelo {model_name}")

        # Carregamos o dicionário de pesos bruto do arquivo
        state_dict = torch.load(f=path)
        
        # Limpando os prefixos indesejados
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Se o PyTorch salvou com o prefixo 'model.', remove os 6 primeiros caracteres
            if key.startswith('model.'):
                new_key = key[6:]
            # Analogo para 'module.', que ocorre quando o modelo foi treinado usando DataParallel ou DistributedDataParallel.
            elif key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
                
            cleaned_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=strict)

        # Verificando se existem chaves faltando ou chaves inesperadas
        if strict == False:
            if len(unexpected_keys) > 0:
                raise RuntimeError("Chaves inesperadas encontradas ao carregar o modelo:\n"
                                  f"{unexpected_keys}\n")
            
            if len(missing_keys) > 0 and ignore_key_name is not None:

                error_missing_keys = []
                
                for key in missing_keys:
                    if not any(ignore_key in key for ignore_key in ignore_key_name):
                        error_missing_keys.append(key)
                
                if len(error_missing_keys) > 0:
                    warnings.warn("Chaves faltando encontradas ao carregar o modelo:\n"
                                  f"{error_missing_keys}\n")

    else:
        print(f"Pesos do modelo {model_name} nao encontrados.")

    return model


def show_netron_model(model_path: str, port: int = 8081) -> None:
    time.sleep(3.)
    netron.start(model_path, address=("localhost", port), browse=False)
    return IFrame(src=f"http://localhost:{port}", width="100%", height="400px")