# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BoomConfig

from transformers import set_seed

set_seed(3)

my_config = BoomConfig()
model_path = "/leonardo_work/IscrB_Decentro/boom_warmup_13B_ckp/hf_permute/step-184000/"
# my_config.save_pretrained(model_path)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # dtype=torch.bfloat16,
    device_map="auto"  # Automatically distribute across available GPUs
)

def test_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
            top_k=50,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test it
prompt = "The boy opened the ancient wooden box, and inside he found"
result = test_model(prompt)

print("")
print(result)
