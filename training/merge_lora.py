import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    tuned = PeftModel.from_pretrained(base, args.adapter)
    merged = tuned.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.output)


if __name__ == "__main__":
    main()
