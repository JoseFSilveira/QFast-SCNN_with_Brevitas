# tqdm.auto
import torch
import torch.nn.functional as F
from pathlib import Path
from config import NUM_CLASSES
from train.train_model import TrainModel


class QATwrapper(torch.nn.Module):
    '''
    Wrapper para o modelo quantizado, com as camadas do Fast-SCNN que foram retiradas do original para facilitar a exportacao para o FINN.
    '''

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def output_upsample(self, output: torch.Tensor, size: list[int]) -> torch.Tensor:
        return F.interpolate(output, size=size, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = list(x.shape[2:]) # Salva o tamanho original da entrada para usar no upsample da saida
        x = self.model(x)
        x = self.output_upsample(x, size) # Redimenciona a saida para o mesmo tamanho da entrada, visto que essa camada foi retirada do modelo quantizado.
        return x

class TrainQuantModel(TrainModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Adaptacoes da classe original para evitar de reescrever o codigo de treinamento completo
        self.quant_model = self.model
        self.model = QATwrapper(self.quant_model)

        # Criando Pasta para salvar Modelo e resultados
        model_path = "./model_weights/quant_params/best_quant_model.pth"
        results_path = "./model_weights/quant_params/best_quant_model_results.pt"
        self.model_path = Path(model_path)
        self.results_path = Path(results_path)

        self.model_path.parent.mkdir(parents=True, exist_ok=True) # Cria a pasta para salvar o modelo, caso ela nao exista
        self.results_path.parent.mkdir(parents=True, exist_ok=True) # Cria a pasta para salvar os resultados, caso ela nao exista