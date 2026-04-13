# tqdm.auto
import torch
from torch import optim
from tqdm.auto import tqdm
from pathlib import Path
from config import NUM_CLASSES

class TrainModel:
  
    def __init__(self, model: torch.nn.Module, loss_fn: callable, optim_fn: callable, metrics: dict,
                 val_to_monitor: str="loss",
                 scheduler_name: str="ReduceLROnPlateau", # Opcoes: "OneCycleLR" ou "ReduceLROnPlateau"
                 max_lr: float=1e-3, # valido apenas para o OneCycleLR, e ignorado caso scheduler_fn seja diferente de "OneCycleLR"
                 epochs: int = 5,
                 accumulation_steps: int = 1, # Numero de batches para acumular os gradientes antes de realizar o passo de otimizacao
                 device: torch.device='cpu') -> None:
    
        self.model = model
        self.val_to_monitor = val_to_monitor
        self.epochs = epochs
        self.device = device
        self.max_lr = max_lr
        self.accumulation_steps = accumulation_steps
        self.num_classes = NUM_CLASSES

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
        
        # Criando Pasta para salvar Modelo e resultados
        model_path = "./model_weights/new_params/best_model.pth"
        results_path = "./model_weights/new_params/best_model_results.pt"
        self.model_path = Path(model_path)
        self.results_path = Path(results_path)

        self.model_path.parent.mkdir(parents=True, exist_ok=True) # Cria a pasta para salvar o modelo, caso ela nao exista
        self.results_path.parent.mkdir(parents=True, exist_ok=True) # Cria a pasta para salvar os resultados, caso ela nao exista


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
            # Reduce LR on Plateau configurado para fine-tuning
            mode = 'min' if self.val_to_monitor=='loss' else 'max' # Se a metrica monitorada for a loss, entao queremos minimizar ela, caso contrario queremos maximizar a metrica
            self.scheduler_fn = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim_fn,
                mode=mode,
                factor=0.5,                  # Reduz o LR em 2x quando a metrica monitorada nao melhorar
                patience=5,                  # Espera 5 epochs sem melhoria para reduzir o LR
            )


    def save_results(self, path: str) -> None:

        # Criando Caminho para Salvar os resultados
        #path = Path(path)
        #file_folder = path.parent
        #file_folder.mkdir(parents=True, exist_ok=True)

        # Salvar resultados
        with open(path, "wb") as f:
            torch.save(self.results, f)


    def save_model(self, path: str, results_path: str=None) -> None:

        # Criando Caminho para Salvar Modelo
        #path = Path(path)
        #file_folder = path.parent
        #file_folder.mkdir(parents=True, exist_ok=True)

        # Salvando o modelo
        model_to_save = getattr(self.model, "_orig_mod", self.model) # Verifica se o modelo tem o atributo _orig_mod, que é criado pelo torch.compile, e salva o modelo original caso ele exista, para evitar problemas de compatibilidade ao carregar o modelo salvo
        torch.save(obj=model_to_save.state_dict(), f=path) # Salva o State Dict do modelo antes do torch.compile, para evitar problemas de compatibilidade

        # Salvando os resultados do treino, caso necesssario
        if results_path is not None:
            self.save_results(results_path)


    def load_best_metric(self, path: str) -> dict:
        
        key_string = f"val_{self.metric_to_monitor}" if self.metric_monitor_index is not None else "val_loss"

        # Carrega o melhor valor da metrica monitorada para metricas de validacao do melhor modelo
        #path = Path(path)
        if path.is_file():
            with open(path, "rb") as f:
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

        for metric_fn in self.metric_fns:
            metric_fn.reset() # Reseta os estados internos de cada metrica, caso existam, para garantir que as metricas sejam calculadas corretamente para o epoch atual

        # Zera o gradiente acumulando antes do loop principal
        self.optim_fn.zero_grad()

        for batch, (X, y) in enumerate(tqdm(dataloader)):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X) # O modelo retorna uma tupla com a saida principal e a saida auxiliar
            batch_loss = self.loss_fn(y_pred, y.long()) # Calculate loss for the current batch

            # Normaliza a loss para acumulacao
            batch_loss = batch_loss / self.accumulation_steps
            batch_loss.backward() # Perform backward pass on the current batch's loss

            # Acumula a loss do batch atual para o valor total de loss do epoch, multiplicando pela accumulation_steps para obter o valor correto da loss acumulada
            train_loss += batch_loss.item() * self.accumulation_steps # Accumulate the scalar value for reporting
            
            if (batch+1) % self.accumulation_steps == 0 or (batch+1) == len(dataloader): # Realiza o passo de otimizacao a cada accumulation_steps batches, ou no ultimo batch do dataloader
                self.optim_fn.step() # Update model parameters based on accumulated gradients
                if self.scheduler_name == "OneCycleLR":
                    self.scheduler_fn.step() # Update learning rate based on OneCycleLR schedule
                self.optim_fn.zero_grad() # Zera os gradientes acumulados para o proximo ciclo de acumulo

            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1).squeeze(dim=1) # Converte as probabilidades em classes preditas e remove a dimensão de canal extra
            for metric_fn in self.metric_fns:
                    metric_fn.update(y_pred_class, y) # Atualiza o estado interno de cada metrica com as classes preditas e as classes verdadeiras do batch atual
        
        train_loss /= len(dataloader)

        # Calcula o valor final de cada metrica para o epoch atual, usando os valores acumulados durante a iteracao sobre os batches do dataloader
        for metric_fn in self.metric_fns:
            metric_values.append(metric_fn.compute().cpu())

        return train_loss, metric_values


    def val_step(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        self.model.eval()
        val_loss = 0
        metric_values = []

        for metric_fn in self.metric_fns:
            metric_fn.reset() # Reseta os estados internos de cada metrica, caso existam, para garantir que as metricas sejam calculadas corretamente para o epoch atual

        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                val_loss += self.loss_fn(y_pred, y.long()).item() # Accumulate the scalar value

                y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1).squeeze(dim=1) # Converte as probabilidades em classes preditas e remove a dimensão de canal extra
                for metric_fn in self.metric_fns:
                    metric_fn.update(y_pred_class, y) # Atualiza o estado interno de cada metrica com as classes preditas e as classes verdadeiras do batch atual

        val_loss /= len(dataloader)

        # Calcula o valor final de cada metrica para o epoch atual, usando os valores acumulados durante a iteracao sobre os batches do dataloader
        for metric_fn in self.metric_fns:
            metric_values.append(metric_fn.compute().cpu())

        return val_loss, metric_values
    

    def __call__(self, train_dataloader: torch.utils.data.DataLoader,
                    val_dataloader: torch.utils.data.DataLoader) -> None:

        # Cria variaveis para salvar melhor iIoU e em qual epoch foi atingido
        best_metric = self.load_best_metric(path=self.results_path) # Carrega o melhor iIoU registrado, caso exista, para comparar com os resultados do treino atual
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
                self.scheduler_fn.step(val_metrics[self.metric_monitor_index]) # usar com ReduceLROnPlateau

            # Imprimindo o que esta acontecendo
            train_metric_string = ""
            val_metric_string = ""
            for i, metric_name in enumerate(self.metric_names):
                if metric_name != "IoU": # apenas para metricas que retornam um valor unico, como o iIoU
                    train_metric_string += f"train_{metric_name}: {train_metrics[i].item():.4f} | " # Formatted to 4 decimal places for readability
                    val_metric_string += f"val_{metric_name}: {val_metrics[i].item():.4f} | " # Formatted to 4 decimal places for readability

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
                    self.save_model(path=self.model_path, results_path=self.results_path) # Salva o modelo e os resultados do treino, caso a metrica seja a melhor registrada

        # Sinaliza fim do Treino
        if model_improved:
            print("Treino do modelo foi finalizado!\n"
                f"O modelo com melhor {self.metric_to_monitor} foi registrado no Epoch {best_epoch+1}.\n"
                f"Esse modelo foi salvo no caminho {self.model_path}")
        else:
            print("Treino do modelo foi finalizado!\n"
                  "Nao foi registrado modelo com melhores resultados que o anteriromente salvo.")