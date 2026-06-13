from pathlib import Path
from eval.eval_model import EvalModel
from models.QFastSCNN import QATwrapper


class EvalQuantModel(EvalModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Adaptacoes da classe original para evitar de reescrever o codigo de avaliacao completo
        self.quant_model = self.model
        self.model = QATwrapper(self.quant_model)