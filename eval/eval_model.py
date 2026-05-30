
import torch
from tqdm.auto import tqdm
from pathlib import Path
from config import NUM_CLASSES


class EvalModel():
    def __init__(self, model: torch.nn.Module, state_dict_path: str=None, results_path: str=None, device: torch.device='cpu'):

        self.model = model
        self.state_dict_path = state_dict_path
        self.results_path = results_path
        self.device = device

        if state_dict_path is not None:
            self.load_state_dict()

        if results_path is not None:
            self.train_results = self.load_results()


    def load_state_dict(self) -> None:

        # Carregando apenas os parametros (state_dict()), pois isso flexibiliza o modelo e evita erros de incompatibilidade com parametros e caminhos do modelo original
        # OBS: torch.load() carrega o modelo inteiro, nao apenas os parametros
        path = Path(self.state_dict_path)
        model_name = path.stem # Extrai o nome do modelo a partir do caminho dado
        if path.is_file():
            print(f"Carregando modelo {model_name}")
            self.model.load_state_dict(torch.load(f=path))
        else:
            print(f"Pesos do modelo {model_name} nao encontrados.")
    

    def load_results(self) -> dict | None:

        path = Path(self.results_path)
        file_name = path.stem # Extrai o nome do arquivo a partir do caminho dado
        if path.is_file():
            print(f"Carregando resultados do modelo {file_name}")
            with open(path, "rb") as f:
                results = torch.load(f, map_location='cpu')

            return results
        
        else:
            print(f"Resultados do modelo {file_name} nao encontrados.")
            return None


    def get_best_results(self, IoU_lables: list=None, train_metrics: bool=False) -> dict:
        best_results = {}

        for metric_name, results in self.train_results.items():
            # Seleciona os dados de valicacao por padrao, cas se a flag train_metrics for True, entao todos os dados serao selecionados, insluindo os de treino
            if train_metrics or metric_name.startswith('val_'):
                results_raw = results[-1] # Pega o valor da metrica na ultima epoca de treinamento, que corresponde ao melhor modelo salvo.

                if '_IoU' in metric_name: # Se a metrica for IoU, entao o resultado eh um tensor com IoU para cada classe
                    results_items = list(results_raw) # Pega os valores do IoU de cada classe na ultima epoca de treinamento, e converte para numero real com
                    if IoU_lables is not None:
                        results_final = {IoU_lables[i]: value.item() for i, value in enumerate(results_items)} # Mapeia os valores do IoU de cada classe para o nome da classe correspondente, usando o dicionário de mapeamento de ids para nomes de classes
                    else:
                        results_final = {f'class_{i}': value.item() for i, value in enumerate(results_items)} # Se o dicionario de mapeamento de ids para nomes de classes não for fornecido

                else:
                    results_final = float(results_raw) # Converte o valor da metrica para numero real com .item() caso nao seja IoU, que possui apenas um numero a representando
                    
                best_results[metric_name] = results_final
        return best_results


    def eval(self, dataloader: torch.utils.data.DataLoader, metrics: dict, loss_fn: callable=None):

        self.model.eval()

        metric_names = list(metrics.keys())
        metric_fns = list(metrics.values()) 

        val_loss = 0.0
        metric_values = []

        for metric_fn in metric_fns:
            metric_fn.reset() # Reseta os estados internos de cada metrica, caso existam, para garantir que as metricas sejam calculadas corretamente para o epoch atual

        with torch.inference_mode():
            for (X, y) in tqdm(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)

                if loss_fn is not None:
                    val_loss += loss_fn(y_pred, y.long()).item() # Accumulate the scalar value

                y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1).squeeze(dim=1) # Converte as probabilidades em classes preditas e remove a dimensão de canal extra
                for metric_fn in metric_fns:
                    metric_fn.update(y_pred_class, y) # Atualiza o estado interno de cada metrica com as classes preditas e as classes verdadeiras do batch atual

        val_loss /= len(dataloader) # Calcula a media da loss para o epoch atual
       
        # Calcula o valor final de cada metrica para o epoch atual, usando os valores acumulados durante a iteracao sobre os batches do dataloader
        for metric_fn in metric_fns:
            metric_values.append(metric_fn.compute().cpu())
            
        # Print reuslts
        results_str = ""
        if loss_fn is not None:
            results_str += f"Loss - {val_loss:.4f} | "
        for i, metric_name in enumerate(metric_names):
            if metric_name != 'IoU': # Se a metrica nao for IoU, entao o valor da metrica eh um numero real representando a media da metrica para o epoch atual
                results_str += f"{metric_name} - {metric_values[i].item():.4f} | "

        print(f"Eval Results: {results_str}")

        return metric_values