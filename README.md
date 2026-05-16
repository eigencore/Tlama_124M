# Tlama (124M) - Compact & Efficient Language Model

<div align="center">
  <img src="https://raw.githubusercontent.com/eigencore/Tlama_124M/refs/heads/main/EC.png" alt="EigenCore" width="150" style="margin-right: 20px;">
  <!--img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="150"-->
</div>

Tlama (124M) is a language model based on **LLaMa (127M)** optimized by **EigenCore**. It is designed for **computational efficiency and scalability**, allowing its use on resource-limited hardware without compromising performance.

This is just the beginning of a development aimed at creating more **competitive and efficient language models**. Future iterations will focus on improving accuracy, adaptability, and hardware efficiency.

## 🚀 Key Features
- **Architecture based on LLaMa (124M)** with efficiency improvements.
- **Trained on the edu_fineweb10B dataset**, a subset of FineWeb with 10 billion tokens.
- **Compatible with Hugging Face `transformers`**.
- **Advanced optimizations:** Flash Attention, Mixed Precision Training, Gradient Clipping, Torch.compile.
- **Trainable on consumer hardware**, such as NVIDIA RTX 4060 GPUs.

## 📌 Using the Model on Hugging Face
You can easily load Tlama (124M) with `transformers`:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("eigencore/tlama-124M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("eigencore/tlama-124M", trust_remote_code=True)

prompt = "Once upon a time in a distant kingdom..."
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 📖 Architecture and Training Data
- **Model:** Based on LLaMa with **124M parameters**.
- **Dataset:** edu_fineweb10B, a web-crawled dataset with 10B tokens.
- **Infrastructure:** Trained on **8 NVIDIA A100 GPUs**, optimized for training on consumer hardware.
- **Optimization Techniques:**
  - **Weight Tying** to reduce parameters.
  - **Flash Attention** for faster inference.
  - **Gradient Accumulation** to train with lower VRAM requirements.
  - **Learning Rate Scheduling** with warmup and cosine decay.

## 📊 Benchmark and Performance
Compared to LLaMa (124M), Tlama (124M):
✅ **Reduces inference time** thanks to computational optimizations.
✅ **Shows competitive performance on language modeling tasks**.
✅ **Enables efficient training on accessible hardware**.

## 📩 Contact & Contributions
This model is developed by **EigenCore**.
- 🌐 More info: [eigencore.org](https://eigencore.org)
- 📧 Contact: max@eigencore.ai
- 🛠️ Contributions: Open to the community!

---
Explore Tlama (124M) and help us build more efficient AI models! 🚀

