# Entity-Aware Medical Image Captioning

A sophisticated deep learning system for generating clinically accurate and entity-aware captions for medical images using transformer-based models and clinical knowledge integration.

## 🎯 Project Overview

This project implements an advanced medical image captioning system that:

- **Extracts clinical entities** from medical image captions using NLP techniques
- **Generates accurate descriptions** of medical images with entity awareness
- **Integrates medical knowledge** through BiomedBERT tokenization and a clinical knowledge engine
- **Analyzes multi-modal medical data** (CT, MRI, X-Ray, Ultrasound, PET scans)
- **Supports modality-specific processing** for different imaging techniques

## 📋 Key Features

### 1. **Clinical Entity Extraction**
- Automatic extraction of medical entities from captions using spaCy NLP
- Linguistic pattern matching (Adjectives + Nouns) for comprehensive entity identification
- High-quality vocabulary filtering with frequency-based pruning
- Handles rare and specialized clinical terminology

### 2. **Medical Domain Knowledge**
- **Clinical Knowledge Engine** with multi-layer architecture:
  - High-speed cache for common radiological findings
  - Medical root-based heuristics for unknown terms
  - WordNet integration for semantic understanding
  - Smart fallback explanations

### 3. **Advanced NLP Processing**
- **PubMedBERT Tokenizer** (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
- Domain-specific biomedical text processing
- Support for complex clinical terminology
- Automatic modality inference from captions

### 4. **Multi-Modal Dataset Support**
- **ROCO Dataset Integration** (Radiology Objects in COntext)
- Support for multiple imaging modalities:
  - CT (Computed Tomography) Scans
  - MRI (Magnetic Resonance Imaging)
  - X-Ray/Radiographs
  - Ultrasound (Sonography)
  - PET Scans
- Stratified sampling for balanced modality coverage

### 5. **Visualization & Analysis**
- Sample visualization across different imaging modalities
- Entity frequency analysis and statistics
- Clinical entity stratification and distribution

## 🚀 Getting Started

### Prerequisites

```
Python 3.8+
PyTorch with CUDA support (recommended)
Google Colab (for cloud execution)
```

### Installation & Setup

1. **Mount Google Drive** (for Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision transformers pandas pillow tqdm matplotlib numpy spacy
   pip install -q kaggle
   python -m spacy download en_core_web_sm
   ```

3. **Setup Kaggle Credentials**:
   - Download your `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/settings/account)
   - Upload to the notebook when prompted
   - System will automatically move it to `~/.kaggle/`

4. **Download ROCO Dataset**:
   ```bash
   kaggle datasets download -d drutikapidikiti/dataset-roco
   unzip dataset-roco.zip -d /content/roco_dataset
   ```

## 📊 Data Processing Pipeline

### Step 1: Dataset Localization
The system automatically detects and catalogs:
- Train/test/validation image directories
- Caption CSV files
- Modality distribution

### Step 2: Entity Extraction
```python
# Automatic extraction of clinical entities
entities = get_comprehensive_entities(caption)
# Example output: ["left parotiditis", "head ct"]
```

### Step 3: Vocabulary Curation
- Filters entities with minimum frequency threshold (≥3 occurrences)
- Removes typos and anomalies
- Creates clean clinical vocabulary

### Step 4: Modality Classification
Infers imaging modality from captions:
- X-Ray/Radiograph detection
- CT Scan identification
- MRI recognition
- Ultrasound/Sonography detection
- PET Scan classification

## 🧠 Model Architecture

### Components

1. **Image Encoder**
   - ResNet-based visual feature extraction
   - Multi-scale feature representation
   - CUDA-optimized processing

2. **Text Encoder/Decoder**
   - PubMedBERT tokenizer for biomedical text
   - Transformer-based sequence generation
   - Entity-aware attention mechanisms

3. **Clinical Knowledge Engine**
   - Four-layer reasoning system:
     1. Cache layer (common findings)
     2. Heuristic rules (medical suffixes)
     3. WordNet integration
     4. Smart fallback mechanisms

4. **Modality Classifier**
   - Multi-class classification (5 modalities)
   - Modality-specific feature processing

## 📈 Dataset Statistics

### ROCO Dataset
- **Total Images**: Thousands of annotated medical images
- **Caption Coverage**: 100% of dataset
- **Modalities**: CT, MRI, X-Ray, Ultrasound, PET, Others
- **Unique Entities**: Typically 5,000+ clinical phrases
- **Average Caption Length**: 15-30 words

### Extracted Vocabulary Example
| Entity | Frequency |
|--------|-----------|
| pneumothorax | 234 |
| pleural effusion | 189 |
| consolidation | 156 |
| cardiomegaly | 128 |
| nodule | 112 |

## 🔍 Key Functions & Classes

### ClinicalKnowledgeEngine
```python
engine = ClinicalKnowledgeEngine()
explanation = engine.get_explanation("pleural effusion")
# Output: "A buildup of excess fluid between the layers of the pleura..."
```

### Entity Extraction
```python
doc = nlp(caption)
entities = get_comprehensive_entities(caption)
```

### Modality Inference
```python
modality = infer_modality(caption)
# Returns: 'X-Ray', 'CT Scan', 'MRI', 'Ultrasound', 'PET Scan', or 'Other'
```

## 📁 Output Files

The notebook generates the following files:

1. **extracted_roco_entities_vocab.csv**
   - All unique entities with frequencies
   - No filtering applied

2. **clean_clinical_vocab.csv**
   - High-quality entities (frequency ≥ 3)
   - Filtered for research use

3. **Model Checkpoints**
   - Saved to: `./medical_transformer_checkpoint.pth`
   - Contains trained weights and architecture

## 🎓 Training & Evaluation

### Model Training
- GPU acceleration with CUDA support
- Mini-batch training with progress tracking
- Checkpoint saving for best models
- Early stopping mechanism

### Evaluation Metrics
- BLEU scores
- METEOR scores
- CIDEr scores
- Entity accuracy (Entity-F1)

## 💡 Usage Examples

### Running the Full Pipeline
```python
# 1. Load dataset
df = pd.read_csv('train_captions.csv')

# 2. Extract entities
df['entities'] = df['caption'].apply(get_comprehensive_entities)

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)

# 4. Visualize samples
display_10_modality_samples(df, file_info)

# 5. Query knowledge engine
explanation = diagnostic_engine.get_explanation("atelectasis")
```

### Inference on New Images
```python
# Load checkpoint
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# Generate caption
with torch.no_grad():
    caption = model.generate(image_tensor, max_length=30)
```

## 🔧 Configuration

Key configuration parameters:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./medical_transformer_checkpoint.pth"
MIN_FREQUENCY = 3  # Minimum entity frequency threshold
VOCAB_SIZE = 28470  # PubMedBERT vocabulary size
```

## 📚 Dataset References

- **ROCO Dataset**: Pelka et al., 2018
  - Paper: "ROCO: Towards a Medical-domain Annotated Image Corpus"
  - Access: [Kaggle ROCO Dataset](https://www.kaggle.com/datasets/drutikapidikiti/dataset-roco)

- **PubMedBERT**: Gu et al., 2021
  - Pretrained on PubMed abstracts and full-text articles
  - Optimized for biomedical NLP tasks

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.x |
| Vision Encoding | torchvision (ResNet) |
| NLP | Hugging Face Transformers |
| Text Processing | spaCy, WordNet |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, PIL |
| Data Download | Kaggle API |
| Cloud Platform | Google Colab |

## ⚡ Performance Optimization

- **GPU Acceleration**: CUDA-enabled tensor operations
- **Mixed Precision**: Optional FP16 training
- **Batch Processing**: Efficient DataLoader implementation
- **Caching**: Multi-layer cache for knowledge engine
- **Modality Stratification**: Balanced sampling strategies

## 🐛 Troubleshooting

### Issue: Dataset not found
**Solution**: Ensure Kaggle credentials are uploaded and dataset download completed successfully

### Issue: Out of memory
**Solution**: Reduce batch size or image resolution in configuration

### Issue: PubMedBERT loading fails
**Solution**: System automatically falls back to BioBERT-v1.1

### Issue: No images displayed
**Solution**: Verify train_dir path and image extensions (.jpg, .jpeg, .png)

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@project{entity_aware_medical_captioning,
  title={Entity-Aware Medical Image Captioning},
  year={2024}
}
```

## 📄 License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Last Updated**: May 2026  
**Status**: Active Development  
**Python Version**: 3.8+  
**PyTorch Version**: 2.0+
