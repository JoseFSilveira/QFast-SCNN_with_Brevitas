# tqdm.auto
import torch
from torch import nn
from torch import optim
from kornia import losses
# Mais informacoes em https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html#torchmetrics.classification.MulticlassJaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex # Jaccard Index eh a mesma coisa que IoU (Intersection over Union) e esse opjeto pode calcular tambem iIoU
from tqdm.auto import tqdm
from pathlib import Path

NUM_CLASSES = 20 # Numero de classes do dataset, incluindo a classe de ignorar (void)

class TrainLinkNet:
  
    def __init__(self, model: torch.nn.Module, loss_fn: callable, optim_fn: callable, metrics: dict,
                 val_to_monitor: str="loss",
                 scheduler_name: str="OneCycleLR",
                 max_lr: float=1e-3, # valido apenas para o OneCycleLR, e ignorado caso scheduler_fn seja diferente de "OneCycleLR"
                 epochs: int = 5,
                 device: torch.device='cpu') -> None:
    
        self.model = model.to(device)
        self.epochs = epochs
        self.device = device
        self.max_lr = max_lr

        valid_scheduler_names = ["OneCycleLR", "ReduceLROnPlateau"]
        if scheduler_name in valid_scheduler_names:
            self.scheduler_name = scheduler_name
        else:
            raise ValueError(f"Scheduler desconhecido: {scheduler_name}. Use algum da lista {valid_scheduler_names}.")

        # Definir loss fuinction e otimizador
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn

        # Definir metricas
        self.metric_names = list(metrics.keys()) # Armazena os nomes das metricas para facilitar a impressao dos resultados do treino
        self.metric_fns = [metric for metric in metrics.values()] # Armazena as funcoes de metricas em uma lista, para facilitar o loop de treino, e move as metricas para o dispositivo correto
        self.metric_types = self.test_metric_output() # Testa se as metricas retornam valores unicos ou tensores com mais de um valor, para ajudar na inicializacao do loop de treino

        # Definir a metrica a ser monitorada para salvar o melhor modelo
        if val_to_monitor in metrics.keys() or val_to_monitor == 'loss':
            self.metric_to_monitor = val_to_monitor
            if val_to_monitor != 'loss':
                self.metric_monitor_index = list(metrics.keys()).index(self.metric_to_monitor) # Encontra o indice da metrica a ser monitorada no dicionario de metricas, para acessar o valor correto na lista de metricas calculadas
            else:
                self.metric_monitor_index = None # Se a metrica a ser monitorada for a loss, nao precisamos de um indice para acessar o valor da metrica, pois ela sera armazenada separadamente
        else:
            raise ValueError(f"Metrica desconhecida: {val_to_monitor}. Use alguma das chaves do dicionario de metricas ou 'loss'.")

        # Criando dicionario para armazenar a evolucao do learning rate
        self.learning_rates = [self.optim_fn.param_groups[0]['lr']] # Armazena o learning rate inicial para analise posterior

        # Criando dicionario para armazenar os resultados
        self.results = {"train_loss": [], "val_loss": []}
        for metric_name in self.metric_names:
            self.results[f"train_{metric_name}"] = []
            self.results[f"val_{metric_name}"] = []
        
        # Criando Pasta para salver Modelo
        self.MODEL_PATH = Path.cwd() / Path("saved_models") # Armazenar modelos salvos numa pasta nova dentro da pasta do projeto
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    

    def get_scheduler(self, dataloader: torch.utils.data.DataLoader):
         # Define o scheduler de learning rate
        if self.scheduler_name == "OneCycleLR":
            self.scheduler_fn = optim.lr_scheduler.OneCycleLR(
                self.optim_fn,
                max_lr=self.max_lr,           # O pico da taxa de aprendizado
                epochs=self.epochs,           # Total de épocas
                steps_per_epoch=len(dataloader), # Quantidade de batches por época
                pct_start=0.3,                # Gasta 30% do treino subindo o LR, e 70% descendo
                div_factor=10.0,              # LR inicial sera max_lr / 10 (ou seja, começa em 1e-4)
                final_div_factor=1000.0       # LR final será muito próximo de zero
        )
            
        elif self.scheduler_name == "ReduceLROnPlateau":
            self.scheduler_fn = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim_fn,
                mode='max',                  # Queremos maximizar a metrica monitorada (iIoU), entao o modo é 'max'
                factor=0.1,                  # Reduz o LR em 10x quando a metrica monitorada nao melhorar
                patience=2,                  # Espera 2 epochs sem melhoria para reduzir o LR
                verbose=True                 # Imprime mensagens quando o LR for reduzido
            )


    def test_metric_output(self) -> list[type]:
        # Testa se as metricas retornam valores unicos ou tensores com mais de um valor, para ajudar na inicializacao do loop de treino
        y_true = torch.randint(low=0, high=NUM_CLASSES, size=(10, 10)).to(self.device)
        y_pred = torch.randint(low=0, high=NUM_CLASSES, size=(10, 10)).to(self.device)

        metric_out_types = []
        for metric_fn in self.metric_fns:
            metric_value = metric_fn(y_pred, y_true)
            
            # Verifica se a saida das metricas eh um tensor, caso contrario o loop de treino sera inicializado incorretamente
            if not isinstance(metric_value, torch.Tensor):
                raise ValueError(f"Valor da metrica {metric_fn} nao eh um tensor. Verifique a implementacao da metrica.")
            
            # Verifica se a saida das metricas eh um tensor escalar (valor unico) ou um tensor com mais de um valor, para ajudar na inicializacao do loop de treino
            if metric_value.shape == torch.Size([]):
                metric_out_types.append(int)
            else:
                metric_out_types.append(list)

        return metric_out_types


    def save_results(self, name: str, best_metric: int=0) -> None:

        # Salvar resultados
        RESULTS_SAVE_PATH = self.MODEL_PATH / Path(f"{name}_results.pt")
        with open(RESULTS_SAVE_PATH, "wb") as f:
            torch.save(self.results, f)


    def save_model(self, name: str ="best_model", save_results: bool=False, best_metric=None) -> None:

        '''
        best_metric sera considerado apenas se save_results for True
        '''

        # Criando Caminho para Salvar Modelo
        MODEL_SAVE_PATH = self.MODEL_PATH / Path(name + ".pth")

        # Salvando o modelo
        torch.save(obj=self.model.state_dict(), f=MODEL_SAVE_PATH) # Salva apenas os parametros do modelo (state_dict)

        # Salvando os resultados do treino, caso necesssario
        if save_results:
            self.save_results(name, best_metric)


    def load_best_metric(self, model_name="best_model") -> dict:
        
        key_string = f"val_{self.metric_to_monitor}" if self.metric_monitor_index is not None else "val_loss"

        # Carrega o melhor valor da metrica monitorada para metricas de validacao do melhor modelo
        SAVED_METRIC_PATH = self.MODEL_PATH / Path(model_name + "_results.pt")
        if SAVED_METRIC_PATH.is_file():
            with open(SAVED_METRIC_PATH, "rb") as f:
                best_model_results = torch.load(f)
                best_metric = best_model_results[key_string][-1] if len(best_model_results[key_string]) > 0 else 0
        else:
            best_metric = 0

        print(f"Melhor valor da metrica monitorada ({self.metric_to_monitor}) registrado no modelo salvo: {best_metric:.4f}\n")
 
        return best_metric
    

    def train_step(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        self.model.train()
        train_loss = 0
        metric_values = []
        
        for metric_type in self.metric_types:
            if metric_type == int:
                metric_values.append(0)
            else:
                metric_values.append(torch.zeros(NUM_CLASSES))

        for batch, (X, y) in enumerate(tqdm(dataloader)):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X) # O modelo retorna uma tupla com a saida principal e a saida auxiliar
            if self.model.aux:
                batch_loss = self.loss_fn(y_pred[0], y.long()) + 0.4 * self.loss_fn(y_pred[1], y.long()) # Calcula a loss total como a soma da loss principal e da loss auxiliar, com um peso de 0.4 para a loss auxiliar, conforme sugerido no artigo original do Fast-SCNN
            else:
                batch_loss = self.loss_fn(y_pred[0], y.long()) # Calculate loss for the current batch
            train_loss += batch_loss.item() # Accumulate the scalar value for reporting

            self.optim_fn.zero_grad()
            batch_loss.backward() # Perform backward pass on the current batch's loss
            self.optim_fn.step()
            if self.scheduler_name == "OneCycleLR":
                self.scheduler_fn.step() # usar com OneCycleLR

            y_pred_class = torch.softmax(y_pred[0], dim=1).argmax(dim=1).squeeze(dim=1) # Converte as probabilidades em classes preditas e remove a dimensão de canal extra
            for i, metric_fn in enumerate(self.metric_fns):
                metric_values[i] += metric_fn(y_pred_class, y).cpu() # Calcula as metricas para o batch atual e acumula os valores em uma lista
        
        train_loss /= len(dataloader)
        for i, metric_value in enumerate(metric_values):
            if self.metric_types[i] == int:
                metric_values[i] = metric_value / len(dataloader) # Calcula a media das metricas para o epoch atual, caso a metrica retorne um valor unico
            else:
                metric_values[i] = torch.clone(metric_value) / len(dataloader) # Calcula a media das metricas para o epoch atual, caso a metrica retorne um tensor com mais de um valor

        return train_loss, metric_values


    def val_step(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        self.model.eval()
        val_loss = 0
        metric_values = []

        for metric_type in self.metric_types:
            if metric_type == int:
                metric_values.append(0)
            else:
                metric_values.append(torch.zeros(NUM_CLASSES))

        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                val_loss += self.loss_fn(y_pred[0], y.long()).item() + 0.4 * self.loss_fn(y_pred[1], y.long()).item() # Accumulate the scalar value

                y_pred_class = torch.softmax(y_pred[0], dim=1).argmax(dim=1).squeeze(dim=1) # Converte as probabilidades em classes preditas e remove a dimensão de canal extra
                for i, metric_fn in enumerate(self.metric_fns):
                    metric_values[i] += metric_fn(y_pred_class, y).cpu() # Calcula as metricas para o batch atual e acumula os valores em uma lista

        val_loss /= len(dataloader)
        for i, metric_value in enumerate(metric_values):
            if self.metric_types[i] == int:
                metric_values[i] = metric_value / len(dataloader) # Calcula a media das metricas para o epoch atual, caso a metrica retorne um valor unico
            else:
                metric_values[i] = torch.clone(metric_value) / len(dataloader) # Calcula a media das metricas para o epoch atual, caso a metrica retorne um tensor com mais de um valor

        return val_loss, metric_values
    

    def __call__(self, train_dataloader: torch.utils.data.DataLoader,
                    val_dataloader: torch.utils.data.DataLoader) -> None:

        # Cria variaveis para salvar melhor iIoU e em qual epoch foi atingido
        best_metric = self.load_best_metric() # Carrega o melhor iIoU registrado, caso exista, para comparar com os resultados do treino atual
        best_epoch = 0

        # Variavel auxiliar para saber se o treino atual registrou um modelo melhor que o anteriormente salvo, caso ele exista.
        model_improved = False

        # Define o scheduler de learning rate
        self.get_scheduler(train_dataloader)

        for epoch in range(self.epochs):
            print(f"EPOCH {epoch+1}/{self.epochs}")

            # Rodar um eoch para todas as fracoes do dataloader
            train_loss, train_metrics = self.train_step(train_dataloader)
            val_loss, val_metrics = self.val_step(val_dataloader)

            # Atualiza o learning rate do otimizador, caso a iIoU de validacao nao tenha melhorado em relacao ao melhor valor registrado, caso o scheduler seja o ReduceLROnPlateau
            if self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(val_metrics[self.metric_monitor_index]) # usar com ReduceLROnPlateau

            # Imprimindo o que esta acontecendo
            train_metric_string = ""
            val_metric_string = ""
            for i, metric_name in enumerate(self.metric_names):
                if self.metric_types[i] == int: # apenas para metricas que retornam um valor unico, como o iIoU
                    train_metric_string += f"train_{metric_name}: {train_metrics[i]:.4f} | " # Formatted to 4 decimal places for readability
                    val_metric_string += f"val_{metric_name}: {val_metrics[i]:.4f} | " # Formatted to 4 decimal places for readability

            print(f"train_loss: {train_loss:.4f} | " # Formatted to 4 decimal places for readability
                    f"{train_metric_string}"
                    f"val_loss: {val_loss:.4f} | "
                    f"{val_metric_string}\n")

            # Armazenando o learning rate atual para analise posterior
            self.learning_rates.append(self.scheduler_fn._last_lr[0])

            # Anexando os resultados do treino e validacao para o dicionario de resultados
            self.results['train_loss'].append(train_loss)
            self.results['val_loss'].append(val_loss)
            for i, metric_name in enumerate(self.metric_names):
                self.results[f'train_{metric_name}'].append(train_metrics[i])
                self.results[f'val_{metric_name}'].append(val_metrics[i])

            # Armazena o modelo caso a metrica monitorada for a maior registrada
            if self.metric_monitor_index is not None: # Se a metrica monitorada nao for a loss, entao comparamos o valor da metrica para decidir se salvamos o modelo ou nao
                if val_metrics[self.metric_monitor_index] > best_metric:
                    model_improved = True
                    best_metric = val_metrics[self.metric_monitor_index].item() # Atualiza o melhor valor da metrica monitorada, convertendo para um valor escalar usando .item()
                    best_epoch = epoch
                    self.save_model(save_results=True, best_metric=best_metric) # Salva o modelo e os resultados do treino, caso a metrica seja a melhor registrada

        # Sinaliza fim do Treino
        if model_improved:
            print("Treino do modelo foi finalizado!\n"
                f"O modelo com melhor {self.metric_to_monitor} foi registrado no Epoch {best_epoch+1}.\n"
                f"Esse modelo foi salvo no caminho {self.MODEL_PATH}")
        else:
            print("Treino do modelo foi finalizado!\n"
                  "Nao foi registrado modelo com melhores resultados que o anteriromente salvo.")