# Parameter-Efficient Fine-Tuning for Portuguese Question Answering

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-blue)](paper/)
[![Models](https://img.shields.io/badge/Models-BERTimbau-green)](https://huggingface.co/neuralmind)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Evaluating LoRA, QLoRA, DoRA and QDoRA on BERTimbau for Portuguese Question Answering**
>
> Mariela M. Nina  
> Universidade Federal de São Paulo (UNIFESP)

## 📄 Paper

This repository contains the code and experiments for the paper:

**"Efficient Fine-Tuning Methods for Portuguese Question Answering: Evaluating LoRA, QLoRA, DoRA and QDoRA on BERTimbau with SQuAD"**

**Key Findings:**
- LoRA achieves **95.8%** of full fine-tuning performance with **73.5%** reduction in training time
- QLoRA enables training BERTimbau-Large on **8GB GPUs** (validated on RTX 2080 Ti)
- Learning rate **2e-4** is critical for PEFT methods (+6.2 F1 points improvement)
- Larger models are more resilient to quantization (4.83 vs 9.56 F1 loss)
- First quantization study applied to BERTimbau models

**Paper and Figures:** See ([paper/](https://github.com/MarielaNina/Projeto_Final_Redes_Neurais/blob/main/Paper_Mariela_Nina.pdf)) directory for the full paper, figures, and supplementary materials.

##  Overview

This project explores **Parameter-Efficient Fine-Tuning (PEFT)** techniques for Portuguese Question Answering using BERTimbau models on the SQuAD v1 dataset translated to Brazilian Portuguese.

### Methods Evaluated
- **LoRA** (Low-Rank Adaptation)
- **QLoRA** (Quantized LoRA with 4-bit NF4)
- **DoRA** (Weight-Decomposed LoRA)
- **QDoRA** (Quantized DoRA)
- **Full Fine-Tuning** (baseline)

### Models
- **BERTimbau-Base** (110M parameters)
- **BERTimbau-Large** (335M parameters)

##  Main Results

### BERTimbau-Large (2 epochs, lr=2e-4)

| Method | F1 | EM | Training Time | Memory |
|--------|-----|-----|--------------|--------|
| Full Fine-Tuning | 84.86 | 73.00 | 5h 15m | 20GB |
| **LoRA** | **81.32** | **68.67** | **1h 24m** | 20GB |
| **QLoRA** | **80.03** | **67.17** | **1h 19m** | **8GB** ✨ |
| DoRA | 80.61 | 68.09 | 1h 48m | 20GB |
| QDoRA | 77.96 | 65.05 | 1h 58m | 8GB |

**QLoRA enables training Large models on consumer GPUs with minimal performance loss!**

### BERTimbau-Base (2 epochs, lr=2e-4)

| Method | F1 | EM | Training Time |
|--------|-----|-----|--------------|
| Full Fine-Tuning | 82.79 | 70.91 | 1h 40m |
| LoRA | 78.01 | 64.85 | 32m |
| QLoRA | 73.23 | 60.26 | 30m |
| DoRA | 78.01 | 64.89 | 40m |
| QDoRA | 74.41 | 61.26 | 42m |

##  Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# OR using conda (recommended)
conda create -n qa_squad python=3.10 -y
conda activate qa_squad
```

### 2. Install Dependencies

```bash
cd qa_bertimbau
pip install -r requirements.txt
```

**Key dependencies:**
- `transformers==4.36.0`
- `peft==0.7.1`
- `bitsandbytes==0.41.0` (for quantization)
- `torch==2.1.0`
- `accelerate`

### 3. Run Training

**LoRA (recommended):**
```bash
cd bertimbau_large
python main_lora.py --learning_rate 2e-4 --num_epochs 2
```

**QLoRA (for 8GB GPUs):**
```bash
python main_qlora.py --learning_rate 2e-4 --num_epochs 2
```

**Full Fine-Tuning (baseline):**
```bash
python main.py --learning_rate 4.25e-5 --num_epochs 2
```


## Experiments Configuration

### Optimal Hyperparameters

**For PEFT methods (LoRA, QLoRA, DoRA, QDoRA):**
- Learning rate: `2e-4` **Critical!**
- Epochs: `2`
- LoRA rank: `16`
- LoRA alpha: `32`
- Batch size: `8` (Large) / `16` (Base)

**For Full Fine-Tuning:**
- Learning rate: `4.25e-5`
- Epochs: `2`
- Batch size: `8` (Large) / `16` (Base)

** Important:** PEFT methods require higher learning rates (2e-4) to achieve optimal performance. Using standard BERT learning rates (4.25e-5) results in severe performance degradation.

### Hardware Requirements

**Tested on:**
- **Primary GPU:** NVIDIA RTX A4500 (20GB VRAM)
- **Secondary GPU:** NVIDIA RTX 2080 Ti (8GB VRAM) - for QLoRA validation
- CUDA 12.2

**Recommended:**
- **8GB VRAM:** QLoRA with Large 
- **16-20GB VRAM:** LoRA with Large 
- **<8GB VRAM:** LoRA with Base 

##  Visualizations

The paper includes detailed visualizations:

1. **Learning Rate Impact** (`lr_comparison.pdf`)
   - Shows dramatic effect of learning rate on PEFT methods
   - Baseline collapse with high LR vs PEFT success

2. **Comprehensive Comparison** (`fig_bertimbau_*_f1_mt_comparison.pdf`)
   - F1 and EM scores across all configurations
   - 4 panels: 2 LRs × 2 epochs
   - Direct visual comparison of all methods

##  Citation

If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{nina2025peft,
  title={Efficient Fine-Tuning Methods for Portuguese Question Answering: 
         Evaluating LoRA, QLoRA, DoRA and QDoRA on BERTimbau with SQuAD},
  author={Nina, Mariela M.},
  booktitle={[Conference Name]},
  year={2025},
  organization={Universidade Federal de São Paulo}
}
```

##  Key References

- **BERTimbau:** Souza et al., BRACIS 2020
- **LoRA:** Hu et al., ICLR 2022
- **QLoRA:** Dettmers et al., NeurIPS 2023
- **DoRA:** Liu et al., ICML 2024
- **SQuAD:** Rajpurkar et al., EMNLP 2016

Full bibliography available in [`paper/references_expanded.bib`](paper/references_expanded.bib)

##  Practical Recommendations

### For 8GB VRAM:
```bash
# Use QLoRA with Large
python main_qlora.py --learning_rate 2e-4 --num_epochs 2
# Expected: F1=80.03 (94.3% of baseline)
```

### For 16-20GB VRAM:
```bash
# Use LoRA with Large (best efficiency/performance)
python main_lora.py --learning_rate 2e-4 --num_epochs 2
# Expected: F1=81.32 (95.8% of baseline, 73.5% time reduction)
```

### For Models <200M parameters:
- Avoid quantization (higher degradation in smaller models)
- Use LoRA without quantization

##  Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python main_qlora.py --batch_size 4

# Or use gradient accumulation
python main_qlora.py --gradient_accumulation_steps 2
```

### Poor Performance with PEFT
```bash
# Make sure you're using high learning rate!
python main_lora.py --learning_rate 2e-4  # NOT 4.25e-5
```

### CUDA/bitsandbytes Issues
```bash
# Reinstall with correct CUDA version
pip uninstall bitsandbytes
pip install bitsandbytes==0.41.0
```

## Reproducibility

All experiments are fully reproducible with provided scripts. Key details:
- Random seeds fixed
- Complete hyperparameter specifications
- Hardware requirements documented
- Software versions pinned in `requirements.txt`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Notes

- Heavy artifacts (models, checkpoints) are excluded via `.gitignore`
- To share trained models, use [Hugging Face Hub](https://huggingface.co/)
- Results and logs are saved in `results/` directory
- All figures are publication-quality PDFs (300 DPI)

## Contact

**Mariela M. Nina**
- 🏛️ Universidade Federal de São Paulo (UNIFESP)
- 📧 mariela.nina@unifesp.br
- 🐙 GitHub: [@MarielaNina](https://github.com/MarielaNina)

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**If you find this work useful, please star the repository!** 

##  Related Resources

- [BERTimbau Models](https://huggingface.co/neuralmind)
- [PEFT Library](https://github.com/huggingface/peft)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

---

**Last Updated:** December 2024
