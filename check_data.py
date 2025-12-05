from datasets import load_dataset

# Try different NER datasets
datasets_to_try = [
    "tner/ontonotes5",  # General NER dataset
    "conll2003",  # Classic NER dataset
    "wikiann",  # Multilingual NER
]

print("Trying various NER datasets to find a working one...\n")

for ds_name in datasets_to_try:
    try:
        print(f"Trying: {ds_name}")
        dataset = load_dataset(ds_name, trust_remote_code=True)
        print(f"  ✓ Success!")
        print(f"  Splits: {dataset.keys()}")
        print(f"  Features: {dataset['train'].features}")
        print(f"  Example: {dataset['train'][0]}")
        print()
        break
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:150]}\n")
