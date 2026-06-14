import torch
from pathlib import Path
import warnings

def load_state_dict(model: torch.nn.Module, path: str, strict: bool = True, ignore_key_name: list=None) -> torch.nn.Module | tuple[torch.nn.Module, dict]:

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