import torch
import tiktoken
import random
import numpy as np
from src.model import RB, RBConfig
import time  # Importar la librería time

# Establecer la semilla para reproducibilidad
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Cargar modelo y pesos
checkpoint_path = "log/model_19072.pt"  # Cambia esto según el nombre de tu checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint['config']
model = RB(config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Cargar el tokenizer
enc = tiktoken.get_encoding("gpt2")

def generate_text(prompt, max_length=29, top_k=50, top_p=0.9, temperature=1.0):
    """Genera texto autocompletando el prompt usando el modelo TinyRB"""
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(tokens)
            logits = logits[:, -1, :]  # Última posición de la secuencia
            logits = logits / temperature  # Ajustar la temperatura
            probs = torch.softmax(logits, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[:, indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            topk_probs, topk_indices = torch.topk(probs, top_k)
            next_token = torch.multinomial(topk_probs, 1)  # Sampling top-k
            next_token = topk_indices.gather(-1, next_token)

            tokens = torch.cat((tokens, next_token), dim=1)
            word = enc.decode([next_token.item()])
            print(word, end='', flush=True)  # Imprimir palabra por palabra
            time.sleep(0.1)  # Pausa para dar la ilusión de generación

            if next_token.item() == enc.eot_token:  # Token de finalización
                break

    return enc.decode(tokens.squeeze(0).tolist())

# Modo interactivo en terminal
print("TinyRB model ready. Type a text and press ENTER to complete it (CTRL+C to exit).")
while True:
    try:
        user_input = input("\nPrompt: ")
        print("Tlama: ", end='')
        output = generate_text(user_input, top_k=50, top_p=0.9, temperature=0.7)
        print("\n")
    except KeyboardInterrupt:
        print("\nExiting...")
        break