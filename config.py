import os
import warnings


# -- WARING CONFIGS --
warnings.filterwarnings('once') # Exibe cada aviso apenas uma vez para evitar poluição do console com mensagens repetidas.


# -- DEFINICAO DE VARIAVEIS DE AMBIENTE --
os.environ['ROCR_VISIBLE_DEVICES'] = '0' # Define quais GPUs AMD estão visíveis para o PyTorch. Use '0' para a primeira GPU, '1' para a segunda, etc.
os.environ['TORCH_USE_HIP_DSA'] = '1' # Habilita o uso de Direct Storage Access (DSA) para melhorar a performance em GPUs AMD.
# Usar apenass para debugging, pois causa problemas de performance e estabilidade em alguns casos.
#os.environ['AMD_SERIALIZE_KERNEL'] = '3' # Habilita a serializacao de kernels para depuração e analise de desempenho.


# -- DEFINICAO DE CONSTANTES --

SCRIPT_MODE = "BOTH" # "TRAIN", "TEST" ou "BOTH"
GENERATE_HISTOGRAM = False

DATA_PATH = '/home/jose-vitor/Documents/Cityscapes_Dataset/fine'

#IM_HEIGHT = 512
#IM_WIDTH = 1024
IM_HEIGHT = 1024
IM_WIDTH = 2048

BATCH_SIZE = 12
NUM_WORKERS = os.cpu_count() // 2
EPOCHS = 100
LEARNING_RATE = 5e-5

NUM_CLASSES = 19 # Numero de classes do dataset, excluindo a classe de ignorar (void)
IGNORE_INDEX = 255 # Valor do pixel para a classe de ignorar (void)