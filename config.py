import os
import warnings


# -- WARING CONFIGS --
warnings.filterwarnings('once') # Exibe cada aviso apenas uma vez para evitar poluição do console com mensagens repetidas.


# -- DEFINICAO DE VARIAVEIS DE AMBIENTE --
os.environ['ROCR_VISIBLE_DEVICES'] = '0' # Define quais GPUs AMD estão visíveis para o PyTorch. Use '0' para a primeira GPU, '1' para a segunda, etc.
os.environ['TORCH_USE_HIP_DSA'] = '1' # Habilita o uso de Direct Storage Access (DSA) para melhorar a performance em GPUs AMD.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Habilita a alocacao de memoria expansivel para melhorar a performance e reduzir a fragmentação de memória em GPUs AMD.

# Usar apenas para debugging, pois causa problemas de performance e estabilidade em alguns casos.
#os.environ['AMD_SERIALIZE_KERNEL'] = '3' # Habilita a serializacao de kernels para depuração e analise de desempenho.

# Usar apenas para gerar o QONNX pelo Brevitas. Desabilitar em qualquer outra circunstancia.
#os.environ['BREVITAS_JIT'] = '0' # Desabilita o Just-In-Time (JIT) do Brevitas, o que pode melhorar a compatibilidade e estabilidade em alguns casos, mas pode reduzir a performance.


# -- DEFINICAO DE CONSTANTES --

SCRIPT_MODE = "BOTH" # "TRAIN", "TEST" ou "BOTH"
GENERATE_HISTOGRAM = False

DATA_PATH = '/home/jose-vitor/Documents/Cityscapes_Dataset/fine'

NUM_WORKERS = os.cpu_count() // 2 # Numero de workers para os dataloaders, ajustado para metade do numero de CPUs disponiveis para evitar sobrecarga do sistema.

# Sizing and cropping configs
#IM_HEIGHT = 512
#IM_WIDTH = 1024
IM_HEIGHT = 1024
IM_WIDTH = 2048
CROP_SIZE = [768, 768] # Tamanho do crop aleatorio aplicado durante a data augmentation, o qual ajuda a reduzir o uso de memoria durante o treinamento do modelo quantizado

# Dataset configs
NUM_CLASSES = 19 # Numero de classes do dataset, excluindo a classe de ignorar (void)
IGNORE_INDEX = 255 # Valor do pixel para a classe de ignorar (void)

# Original model Training Hyperparameters
BATCH_SIZE = 12
EPOCHS = 100
LEARNING_RATE = 5e-5

# Quantized model training hyperparameters
BIT_WIDTH = 8
QAT_BATCH_SIZE = 12 # Batch size menor ou igual ao original para o treinamento do modelo quantizado, para evitar problemas de memoria. Ajuste conforme a capacidade da sua GPU.
QAT_EPOCHS = 50 # Treinar por menos epocas do que o modelo original, pois o modelo quantizado tem menos capacidade e pode convergir mais rapido
QAT_LEARNING_RATE = 5e-5