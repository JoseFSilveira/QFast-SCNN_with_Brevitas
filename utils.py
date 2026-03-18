import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from pathlib import Path
# Mais informacoes em https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html#torchmetrics.classification.MulticlassJaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex

import custom_cityscapes as ccs

def plot_leaning_rate_evolution(learning_rates: list[float]) -> None:
    plt.figure(figsize=(5, 10))
    plt.plot(learning_rates)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    x_ticks_interval = len(learning_rates) // 10
    plt.xticks(range(1, len(learning_rates) + 1, x_ticks_interval))
    plt.grid(True, which='major')
    plt.title("Evolucao do Learning Rate ao longo das Epochs")
    plt.show()


def get_best_results(results: dict, IoU_lables: list=None, train_metrics: bool=False) -> dict:
    best_results = {}
    for metric_name, results in results.items():
        if train_metrics or metric_name.startswith('val_'):
            results_raw = results[-1] # Pega o valor da metrica na ultima epoca de treinamento, que corresponde ao melhor modelo salvo.

            if metric_name == 'val_IoU':
                results_items = list(results_raw) # Pega os valores do IoU de cada classe na ultima epoca de treinamento, e converte para numero real com
                if IoU_lables is not None:
                    results_final = {IoU_lables[i]: value.item() for i, value in enumerate(results_items)} # Mapeia os valores do IoU de cada classe para o nome da classe correspondente, usando o dicionário de mapeamento de ids para nomes de classes
                else:
                    results_final = {f'class_{i}': value.item() for i, value in enumerate(results_items)} # Se o dicionario de mapeamento de ids para nomes de classes não for fornecido


            else:
                results_final = float(results_raw) # Converte o valor da metrica para numero real com .item() caso nao seja IoU, que possui apenas um numero a representando
                
            best_results[metric_name] = results_final
    return best_results


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

    for i in range(n):

        axes[i, 0].imshow(imgs[i].permute(1,2,0)) # permute para mudar a ordem dos canais e converter um tensor para imagem
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


def load_results(path: str) -> dict | None:

    path = Path(path)
    file_name = path.stem # Extrai o nome do arquivo a partir do caminho dado
    if path.is_file():
        print(f"Carregando resultados do modelo {file_name}")
        with open(path, "rb") as f:
            results = torch.load(f, map_location='cpu')

        return results
    
    else:
        print(f"Resultados do modelo {file_name} nao encontrados.")
        return None


def load_state_dict(model: torch.nn.Module, path: str, results_path: str=None) -> torch.nn.Module | tuple[torch.nn.Module, dict]:
    
    # Carregando apenas os parametros (state_dict()), pois isso flexibiliza o modelo e evita erros de incompatibilidade com parametros e caminhos do modelo original
    # OBS: torch.load() carrega o modelo inteiro, nao apenas os parametros
    path = Path(path)
    model_name = path.stem # Extrai o nome do modelo a partir do caminho dado
    if path.is_file():
        print(f"Carregando modelo {model_name}")
        model.load_state_dict(torch.load(f=path))
    else:
        print(f"Modelo {model_name} nao encontrado.")

    # Se load_results nao for nulo, entao o modelo e carregado e testado, e os resultados sao impressos, caso contrario, apenas o modelo e carregado e retornado
    if results_path is not None:
        model_results = load_results(results_path)
        return model, model_results
    
    else:
        return model