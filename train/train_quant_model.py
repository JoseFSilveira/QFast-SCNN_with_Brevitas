# tqdm.auto
from pathlib import Path
from train.train_model import TrainModel
from models.QFastSCNN import QATwrapper


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