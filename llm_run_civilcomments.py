
import transformers.utils


if not hasattr(transformers.utils, "LossKwargs"):
    class LossKwargs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    transformers.utils.LossKwargs = LossKwargs
    print("✅ LossKwargs manually patched.")

import os
import time
import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION - CHANGE MODEL_NAME FOR EACH RUN
# ============================================================

# Choose one model per run:
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
#MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
# MODEL_NAME = "google/gemma-3-12b-it"

# Short name for output file
MODEL_SHORT_NAMES = {
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral_7b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_8b",
    "tiiuae/Falcon3-7B-Instruct": "falcon3",
    "microsoft/Phi-3.5-mini-instruct": "phi3.5_mini",
    "google/gemma-3-12b-it": "gemma3_12b",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- DATA CONFIGURATION ----
MAX_EXAMPLES = None  # Max samples to process (None for all)
SPLIT = "train"       # split to use: "train" or "test"
DATA_DIR = "./data"   # Where to download WILDS data

# ---- OUTPUT CONFIGURATION ----
OUTPUT_DIR = "./civilcomments_results"
MAX_PROMPT_LEN = 1024

# ---- CHECKPOINT CONFIGURATION ----
CHECKPOINT_EVERY = 1000  # Save checkpoint every N samples
CHECKPOINT_DIR = "./civilcomments_checkpoints"

# ---- CivilComments Group Information ----
GROUP_NAMES = [
    "male",
    "female",
    "LGBTQ",
    "Christian",
    "Muslim",
    "other_religions",
    "Black",
    "White"
]

# Binary classification labels
LABEL_NAMES = ["non-toxic", "toxic"]
CHOICE_LETTERS = ["A", "B"]


# ============================================================
# STEP 1: Download and Load CivilComments
# ============================================================

def install_wilds():
    """Install WILDS package if not available."""
    try:
        import wilds
        print(f"WILDS version: {wilds.__version__}")
    except ImportError:
        print("Installing WILDS...")
        os.system("pip install wilds --break-system-packages --quiet")
        import wilds
        print(f"WILDS installed: {wilds.__version__}")


def load_civilcomments_data(data_dir, split, max_examples=None):
    """
    Load CivilComments dataset from WILDS.

    Returns:
        texts: List of comment strings
        labels: numpy array of binary labels (0=non-toxic, 1=toxic)
        groups: numpy array of group IDs (0-7)
    """
    from wilds import get_dataset

    print(f"\nDownloading/Loading CivilComments dataset...")
    dataset = get_dataset(dataset="civilcomments", download=True, root_dir=data_dir)

    print(f"Getting {split} split...")
    subset = dataset.get_subset(split)

    n_total = len(subset)
    n_to_load = min(n_total, max_examples) if max_examples else n_total

    print(f"Total samples in {split}: {n_total:,}")
    print(f"Loading: {n_to_load:,} samples")

    texts = []
    labels = []
    groups = []

    for i in tqdm(range(n_to_load), desc="Loading data"):
        text, label, metadata = subset[i]

        texts.append(text)
        labels.append(int(label.item()) if hasattr(label, 'item') else int(label))

        # Extract group from metadata
        if hasattr(metadata, 'numpy'):
            meta_np = metadata.numpy()
        elif hasattr(metadata, 'tolist'):
            meta_np = np.array(metadata.tolist())
        else:
            meta_np = np.array(metadata)

        # Find primary group (first non-zero indicator)
        group_id = 0  # default to first group
        for idx in range(min(8, len(meta_np))):
            if idx < len(meta_np) and meta_np[idx] > 0:
                group_id = idx
                break

        groups.append(group_id)

    labels = np.array(labels)
    groups = np.array(groups)

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(texts):,}")
    print(f"\n  Label distribution:")
    for l in range(2):
        count = np.sum(labels == l)
        pct = count / len(labels) * 100
        print(f"    {LABEL_NAMES[l]}: {count:,} ({pct:.1f}%)")

    print(f"\n  Group distribution:")
    for g in range(8):
        count = np.sum(groups == g)
        pct = count / len(groups) * 100
        print(f"    {GROUP_NAMES[g]}: {count:,} ({pct:.1f}%)")

    
    invalid_indices = []
    for i, text in enumerate(texts):
        if not is_valid_text(text):
            invalid_indices.append(i)

    if invalid_indices:
        print(f"\n⚠️  WARNING: Found {len(invalid_indices)} invalid text entries at indices:")
        print(f"    First 10: {invalid_indices[:10]}")
        print(f"    These will be skipped during processing.")

    return texts, labels, groups


def is_valid_text(text):
    """Check if text is valid for processing."""
    if text is None:
        return False
    if pd.isna(text):
        return False
    if not isinstance(text, str):
        return False
    if len(text.strip()) == 0:
        return False
    return True




def get_checkpoint_path(model_short):
    """Get checkpoint file path for current model."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_{model_short}.json")


def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            print(f"\n✅ Found checkpoint at index {checkpoint['last_idx']}")
            print(f"   Accuracy so far: {checkpoint['correct']}/{checkpoint['processed']} = {checkpoint['correct']/checkpoint['processed']*100:.1f}%")
            return checkpoint
        except (json.JSONDecodeError, KeyError) as e:
            print(f"\n⚠️  Checkpoint file corrupted, starting fresh: {e}")
            return None
    return None


def save_checkpoint(checkpoint_path, last_idx, results, correct, processed,
                   group_correct, group_total, skipped_indices):
    """Save checkpoint to disk."""
    # Convert all numpy types to Python native types for JSON serialization
    checkpoint = {
        'last_idx': int(last_idx),
        'results': results,  # Already converted in result_data
        'correct': int(correct),
        'processed': int(processed),
        'group_correct': {int(k): int(v) for k, v in group_correct.items()},
        'group_total': {int(k): int(v) for k, v in group_total.items()},
        'skipped_indices': [int(i) for i in skipped_indices],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

   
    temp_path = checkpoint_path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f)
    os.replace(temp_path, checkpoint_path)




def load_model_and_tokenizer(model_name):
    """Load HuggingFace model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\nLoading model: {model_name}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate dtype
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(DEVICE)

    model.eval()
    print("Model loaded successfully!")

    return model, tokenizer


def compute_logprob_of_continuation(model, tokenizer, prompt_ids, cont_ids, device):
    """Compute log-probability of continuation tokens given prompt."""
    full_ids = torch.cat([prompt_ids, cont_ids], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(full_ids, use_cache=False)
        logits = outputs.logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)

    prompt_len = prompt_ids.shape[0]
    total_logprob = 0.0

    for i in range(cont_ids.shape[0]):
        logit_pos = prompt_len - 1 + i
        token_id = int(cont_ids[i])
        if logit_pos < 0 or logit_pos >= log_probs.shape[1]:
            return -1e9
        total_logprob += log_probs[0, logit_pos, token_id].item()

    return total_logprob


def tokenize_continuation_for_letter(tokenizer, letter):
    """Return token ids for JSON response with given letter."""
    continuation_string = f'{{"answer": "{letter}"}}'
    ids = tokenizer(continuation_string, add_special_tokens=False).input_ids
    return torch.tensor(ids, dtype=torch.long) if ids else torch.tensor([], dtype=torch.long)


def process_civilcomments(model, tokenizer, texts, labels, groups, output_file,
                         max_prompt_len=1024, checkpoint_every=1000):
    """
    Process CivilComments samples and compute probability vectors.

    Features:
    - Checkpointing for resume capability
    - Handles invalid/NaN text entries
    - Robust error handling
    """

    model_short = MODEL_SHORT_NAMES.get(MODEL_NAME, MODEL_NAME.split("/")[-1])
    checkpoint_path = get_checkpoint_path(model_short)

    # Try to load existing checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint:
        start_idx = checkpoint['last_idx'] + 1
        all_results = checkpoint['results']
        correct = checkpoint['correct']
        processed = checkpoint['processed']
        group_correct = {int(k): v for k, v in checkpoint['group_correct'].items()}
        group_total = {int(k): v for k, v in checkpoint['group_total'].items()}
        skipped_indices = checkpoint['skipped_indices']
        print(f"\n🔄 Resuming from index {start_idx}...")
    else:
        start_idx = 0
        all_results = []
        correct = 0
        processed = 0
        group_correct = {g: 0 for g in range(8)}
        group_total = {g: 0 for g in range(8)}
        skipped_indices = []
        print(f"\n🆕 Starting fresh from index 0...")

    # Formatted choices for prompt
    formatted_choices = "\n".join([
        f"{CHOICE_LETTERS[i]}. {LABEL_NAMES[i]}"
        for i in range(len(CHOICE_LETTERS))
    ])

    start_time = time.time()
    n_examples = len(texts)

    print(f"\nProcessing {n_examples:,} comments (starting from {start_idx})...")

    pbar = tqdm(range(start_idx, n_examples), initial=start_idx, total=n_examples, desc="Processing")

    for idx in pbar:
        text = texts[idx]
        label = labels[idx]
        group = groups[idx]

        # === VALIDATE TEXT ===
        if not is_valid_text(text):
            skipped_indices.append(idx)
            pbar.set_postfix({'skipped': len(skipped_indices), 'acc': f'{correct/max(processed,1)*100:.1f}%'})
            continue

        gt_label_str = LABEL_NAMES[label]
        gt_letter = CHOICE_LETTERS[label]

        # Truncate very long comments
        if len(text) > 1500:
            text_truncated = text[:1500] + "..."
        else:
            text_truncated = text

        # Build prompt using chat template
        messages = [
            {
                "role": "system",
                "content": "You are an expert at detecting toxic comments online. "
                          "Classify whether the following comment is toxic or non-toxic. "
                          "A toxic comment is rude, disrespectful, or likely to make someone leave a discussion. "
                          'Respond with ONLY: {"answer": "X"} where X is A or B.'
            },
            {
                "role": "user",
                "content": f"Comment:\n{text_truncated}\n\nChoices:\n{formatted_choices}\n\nAnswer:"
            },
        ]

        try:
            prompt_tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )[0]
        except Exception as e:
            # Fallback for models without chat template
            prompt_text = f"""You are an expert at detecting toxic comments online.
Classify whether the following comment is toxic or non-toxic.
A toxic comment is rude, disrespectful, or likely to make someone leave a discussion.
Respond with ONLY: {{"answer": "X"}} where X is A or B.

Comment:
{text_truncated}

Choices:
{formatted_choices}

Answer:"""
            prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids[0]

        # Truncate if too long
        if prompt_tokens.shape[0] > max_prompt_len - 32:
            prompt_tokens = prompt_tokens[-(max_prompt_len - 32):]

        # === COMPUTE LOG-PROBABILITIES WITH ERROR HANDLING ===
        try:
            per_letter_logprobs = []
            for letter in CHOICE_LETTERS:
                cont_ids = tokenize_continuation_for_letter(tokenizer, letter)
                lp = compute_logprob_of_continuation(model, tokenizer, prompt_tokens, cont_ids, DEVICE)
                per_letter_logprobs.append(lp)
        except Exception as e:
            print(f"\n⚠️  Error at index {idx}: {e}")
            skipped_indices.append(idx)
            continue

        # Get prediction
        pred_index = int(np.argmax(per_letter_logprobs))
        pred_letter = CHOICE_LETTERS[pred_index]
        pred_label_str = LABEL_NAMES[pred_index]

        processed += 1
        is_correct = (pred_index == label)
        if is_correct:
            correct += 1

        # Track per-group accuracy
        group_total[group] += 1
        if is_correct:
            group_correct[group] += 1

        # Normalize to probabilities
        logprobs_np = np.array(per_letter_logprobs)
        probs_np = np.exp(logprobs_np - logprobs_np.max())
        probs_np = probs_np / probs_np.sum()

        # Update progress bar
        pbar.set_postfix({'acc': f'{correct/processed*100:.1f}%', 'skipped': len(skipped_indices)})

        # Progress logging
        if (idx + 1) % 500 == 0 or idx == start_idx:
            acc_so_far = correct / processed * 100
            print(f"\n[{idx+1}/{n_examples}] Accuracy so far: {acc_so_far:.1f}%")
            print(f"  GT={gt_letter}({gt_label_str}) PRED={pred_letter}({pred_label_str})")
            print(f"  Probs: non-toxic={probs_np[0]:.3f}, toxic={probs_np[1]:.3f}")
            print(f"  Skipped: {len(skipped_indices)} invalid entries")

        
        result_data = {
            "index": int(idx),
            "text": text[:500] + "..." if len(text) > 500 else text,
            "full_text_length": int(len(text)),
            "ground_truth_label": gt_label_str,
            "ground_truth_int": int(label),
            "group_id": int(group),
            "group_name": GROUP_NAMES[int(group)],
            "prediction_label": pred_label_str,
            "prediction_int": int(pred_index),
            "is_correct": bool(is_correct),
            "probabilities": {
                LABEL_NAMES[i]: float(probs_np[i])
                for i in range(len(LABEL_NAMES))
            },
            "log_probabilities": {
                LABEL_NAMES[i]: float(per_letter_logprobs[i])
                for i in range(len(LABEL_NAMES))
            },
        }
        all_results.append(result_data)

        # === SAVE CHECKPOINT ===
        if (idx + 1) % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path, idx, all_results, correct, processed,
                group_correct, group_total, skipped_indices
            )
            print(f"\n💾 Checkpoint saved at index {idx+1}")

    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    overall_acc = correct / processed * 100 if processed > 0 else 0.0

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Processed: {processed:,} comments in {total_time:.1f} seconds")
    print(f"Skipped: {len(skipped_indices)} invalid entries")
    print(f"Throughput: {processed/total_time:.2f} comments/sec")
    print(f"\nOverall Accuracy: {correct}/{processed} = {overall_acc:.2f}%")

    print(f"\nPer-Group Accuracy:")
    group_accs = {}
    for g in range(8):
        if group_total[g] > 0:
            acc = group_correct[g] / group_total[g] * 100
            group_accs[GROUP_NAMES[g]] = acc
            print(f"  {GROUP_NAMES[g]:<20}: {group_correct[g]:>5}/{group_total[g]:<5} = {acc:>6.2f}%")
        else:
            group_accs[GROUP_NAMES[g]] = None
            print(f"  {GROUP_NAMES[g]:<20}: No samples")

    # Worst-group accuracy
    valid_accs = [a for a in group_accs.values() if a is not None]
    worst_group_acc = min(valid_accs) if valid_accs else 0.0
    worst_group_name = [name for name, acc in group_accs.items() if acc == worst_group_acc][0]

    print(f"\n*** WORST-GROUP ACCURACY: {worst_group_acc:.2f}% ({worst_group_name}) ***")
    print("=" * 70)

    # Save final results
    output_data = {
        "metadata": {
            "model": MODEL_NAME,
            "model_short": MODEL_SHORT_NAMES.get(MODEL_NAME, "unknown"),
            "dataset": "civilcomments",
            "split": SPLIT,
            "num_examples": processed,
            "num_skipped": len(skipped_indices),
            "skipped_indices": skipped_indices,
            "overall_accuracy": overall_acc,
            "worst_group_accuracy": worst_group_acc,
            "worst_group_name": worst_group_name,
            "group_accuracies": group_accs,
            "total_time_seconds": total_time,
            "group_names": GROUP_NAMES,
            "label_names": LABEL_NAMES,
        },
        "results": all_results
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file removed (processing complete)")

    return all_results, overall_acc, worst_group_acc



def main():
    print("=" * 70)
    print("CivilComments - LLM Probability Vector Extraction (ROBUST)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Split: {SPLIT}")
    print(f"Max examples: {MAX_EXAMPLES}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY} samples")
    print("=" * 70)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Generate output filename
    model_short = MODEL_SHORT_NAMES.get(MODEL_NAME, MODEL_NAME.split("/")[-1])
    output_file = os.path.join(OUTPUT_DIR, f"civilcomments_{model_short}.json")
    checkpoint_path = get_checkpoint_path(model_short)

    # Check if checkpoint exists (auto-resume)
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Checkpoint found! Will resume from last position.")
        print(f"   Checkpoint file: {checkpoint_path}")
    elif os.path.exists(output_file):
        print(f"\nOutput file already exists: {output_file}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            return

    # Install WILDS if needed
    install_wilds()

    # Load data
    texts, labels, groups = load_civilcomments_data(
        data_dir=DATA_DIR,
        split=SPLIT,
        max_examples=MAX_EXAMPLES
    )

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    
    results, overall_acc, worst_group_acc = process_civilcomments(
        model, tokenizer, texts, labels, groups, output_file,
        MAX_PROMPT_LEN, CHECKPOINT_EVERY
    )

    print("\n" + "=" * 70)
    print("✅ DONE!")
    print(f"Output saved to: {output_file}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Worst-Group Accuracy: {worst_group_acc:.2f}%")
    print("=" * 70)

    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()