# Malayalam-to-English Translation

## Overview
This project is a Python-based application that translates text from Malayalam to English using a fine-tuned transformer model hosted on Hugging Face. It leverages advanced NLP techniques to ensure precise and contextual translations.

## Features
- **Accurate Translations**: Utilizes a fine-tuned transformer model for reliable results.
- **Device Compatibility**: Automatically uses GPU for faster performance if available.
- **Simple to Use**: Provides a straightforward function for generating translations.

## Technologies Used
- **Python**: Core programming language for the implementation.
- **Hugging Face Transformers**: For loading and using the pre-trained model.
- **Torch**: For tensor computations and device handling.


## Example Code to use my model

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer from Hugging Face Hub
model_name = "athul-krishna/malayalam-english"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def translate_new_input(model, tokenizer, input_text, max_length=128):
    # Get the device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    
    # Generate the translation
    outputs = model.generate(inputs["input_ids"], max_length=max_length)
    
    # Decode the generated output
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Example usage
new_input = "റോസാദളങ്ങളാൽ പൊതിഞ്ഞ ഒരു ശരീരം നിങ്ങളുടെ വായ പൂവിട്ടത് എവിടെയാണെന്ന് അടയാളപ്പെടുത്തുന്നു"  # Malayalam input
translated_output = translate_new_input(model, tokenizer, new_input)
print(f"Input: {new_input}")
print(f"Translated Output: {translated_output}")


