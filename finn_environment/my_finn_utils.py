import torch
from torchvision.datasets import Cityscapes
from pathlib import Path
import warnings
import json

from config import IGNORE_INDEX, NUM_CLASSES


def read_json_dict(filename):
    with open(filename, "r") as f:
        ret = json.load(f)
    return ret


def pynqz2_viability_check(resource_dict: dict) -> None:
    '''
    Reads the resource dict and checks if the model can be deployed on the Pynq-Z2 board.
    '''
    # Pynx-Z2 board specs
    pynqz2_limits = {
        "LUT": 53200.0,
        "BRAM_18K": 280.0, # Xilinx list 140 BRAM_36K, which is equivalent to 280 BRAM_18K
        "DSP": 220.0,
        "URAM": 0.0        # The Pynq-Z2 board does not have URAM, so the limit is 0
    }

    totals = resource_dict.get("total", {})
    report_text = f"Pynq-Z2 uses ZYNQ XC7Z020-1CLG400C SoC\nResource usage totals: {totals}\n\n"

    for resource, limit in pynqz2_limits.items():
        usage = totals.get(resource, 0.0) # if no resource is found, assume 0 usage
        usage_percentage = (usage / limit * 100) if limit > 0 else 0.0
        if limit == 0 and usage > 0:
            status_text = "IMPOSSIBLE (resource not available on Pynq-Z2)"
        elif usage_percentage < 115: # Allowing a 15% margin for error since Vivado optimizes hte LUTs
            status_text = "PASS"
        else:
            status_text = "FAIL"
        report_text += f"Checking {resource}: {usage} used, {limit} limit. Usage percentage: {usage_percentage:.2f}%. {status_text}\n"
    print(report_text)



def generate_cityscapes_labels():
    '''
    Function created based on the __init__() function of class CityscapesLables from train_environment/custom_cityscapes.py
    '''

    # Criando listas dos nomes e das cores das classes treinaveis
    id_names = {}
    color_list = []
    lable_conversion = {}
    for c in Cityscapes.classes:

        # Adicionando valores ao dicionario de conversao de ids
        lable_conversion[c.id] = c.train_id if c.train_id != -1 else IGNORE_INDEX # A classe 'ignore' tem train_id -1, entao atribui o valor IGNORE_INDEX para ela
        # Adicionando valores as listas de nomes e cores
        if c.train_id != -1 and c.train_id != 255:
            id_names[c.train_id] = c.name
            color_list.append(c.color)

    # Variavel para dicionario de nomes
    id_names.update({IGNORE_INDEX: 'ignore'}) # Adiciona a classe 'ignore' com train_id 255

    return lable_conversion, id_names


def load_state_dict(model: torch.nn.Module, path: str, strict: bool = True, ignore_key_name: list=None) -> torch.nn.Module | tuple[torch.nn.Module, dict]:
    '''
    Function copied from train_environment/utils.py
    '''
    # Carregando apenas os parametros (state_dict()), pois isso flexibiliza o modelo e evita erros de incompatibilidade com parametros e caminhos do modelo original
    # OBS: torch.load() carrega o modelo inteiro, nao apenas os parametros
    path = Path(path)
    model_name = path.stem # Extrai o nome do modelo a partir do caminho dado

    if path.is_file():
        print(f"Carregando modelo {model_name}")

        # Carregamos o dicionário de pesos bruto do arquivo
        state_dict = torch.load(f=path, map_location='cpu')
        
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


class IdToTrainIdTransform:
    '''
    Class copied from train_environment/custom_transforms.py
    '''
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