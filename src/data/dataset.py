import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from torch.utils.data import DataLoader
from datasets import get_dataset_config_names, load_dataset
import random
import re

def split_thought_solution(text: str):
    thought_re = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
    solution_re = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)

    thought = thought_re.search(text).group(1).strip()
    solution = solution_re.search(text).group(1).strip()

    return thought, solution

def _get_dist_info():
    import os
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size

def _shard_list_contiguous(xs, rank, world_size):
    n = len(xs)
    start = (n * rank) // world_size
    end = (n * (rank + 1)) // world_size
    return xs[start:end]

def _maybe_shard(xs, dont_shard=False):
    rank, world_size = _get_dist_info()
    if world_size <= 1 or dont_shard:
        return xs
    return _shard_list_contiguous(xs, rank, world_size)

def make_concat_chunks(data, tokenizer, max_length, num_samples, seed=0, add_eos=False):
    if num_samples <= 0:
        return []

    eos_id = tokenizer.eos_token_id
    token_buffer = []
    chunks = []
    for ex in data:
        ids = tokenizer(ex["text"], return_tensors=None)["input_ids"]
        if add_eos and eos_id is not None:
            ids = ids + [eos_id]
        token_buffer.extend(ids)

        while len(token_buffer) >= max_length:
            chunk = token_buffer[:max_length]
            del token_buffer[:max_length]
            chunks.append(torch.tensor(chunk, dtype=torch.long))
            if len(chunks) >= num_samples:
                return chunks

    return chunks


def make_message_or_text_chunks(data, tokenizer, max_length, num_samples, seed=0, add_eos=False):
    renderer = _open_thoughts_renderer(tokenizer)

    def examples():
        for ex in data:
            if ex.get("messages") and renderer[0] != "plain_text":
                yield {"text": _render_open_thoughts_messages(ex["messages"], tokenizer, renderer)}
            else:
                yield {"text": ex["text"]}

    return make_concat_chunks(examples(), tokenizer, max_length, num_samples, seed=seed, add_eos=add_eos)

def tokenize_texts(data, tokenizer, max_length, num_samples, seed=0):
    random.seed(seed)
    tokenized_batches = []

    for i, text in enumerate(data):
        if len(tokenized_batches) >= num_samples:
            break
        enc = tokenizer(text["text"], return_tensors="pt")
        if enc.input_ids.shape[1] >= max_length + 1:
            start = random.randint(0, enc.input_ids.shape[1] - max_length - 1)
            end = start + max_length
            tokenized_batches.append(enc.input_ids[:, start:end].squeeze(0))

    print("number of samples:", len(tokenized_batches))
    return tokenized_batches

def c4(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
       shuffle_seed=1234, **_kw):
    train_data = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
        split='train'
    )
    val_data = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
        split='validation'
    )

    train_tokens = tokenize_texts(train_data, tokenizer, max_length, train_samples + gpt_samples, seed=shuffle_seed)
    val_tokens = tokenize_texts(val_data, tokenizer, max_length, val_samples)

    train_loader = DataLoader(train_tokens[:train_samples], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_tokens, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(train_tokens[train_samples:], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, gpt_loader

def fineweb_edu(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
                shuffle_seed=1234, shuffle_buffer_size=100_000, seed=42, **_kw):
    all_data = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        "sample-10BT",
        split='train',
        streaming=True
    )

    all_data = all_data.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    
    all_tokens = make_concat_chunks(all_data, tokenizer, max_length, train_samples + val_samples + gpt_samples, seed=shuffle_seed)

    train_split = _maybe_shard(all_tokens[:train_samples])
    val_split = _maybe_shard(all_tokens[train_samples:train_samples+val_samples])
    gpt_split = _maybe_shard(all_tokens[train_samples+val_samples:])

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(gpt_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, gpt_loader

def _broadcast_chunks(chunks, rank, world_size):
    """Broadcast pre-tokenized chunks from rank 0 to all ranks."""
    import torch.distributed as dist
    if world_size <= 1:
        return chunks

    if rank == 0:
        stacked = torch.stack(chunks)
    else:
        stacked = None

    n = torch.tensor(len(chunks) if rank == 0 else 0, dtype=torch.long, device="cuda")
    dist.broadcast(n, src=0)
    seq_len = torch.tensor(chunks[0].shape[0] if rank == 0 and len(chunks) > 0 else 0,
                           dtype=torch.long, device="cuda")
    dist.broadcast(seq_len, src=0)

    if rank != 0:
        stacked = torch.empty((n.item(), seq_len.item()), dtype=torch.long, device="cuda")
    else:
        stacked = stacked.to("cuda")

    dist.broadcast(stacked, src=0)
    return [stacked[i].cpu() for i in range(stacked.shape[0])]


def _patch_reasoning_chat_template(tokenizer):
    tmpl = getattr(tokenizer, "chat_template", None)
    if not tmpl:
        return False
    tokenizer.chat_template = tmpl.replace(
        "<think></think>{{render_content(message)}}",
        "{%- set rc = message.get('reasoning_content', '') -%}"
        "<think>{{rc}}</think>{{render_content(message)}}"
    )
    return True

def _open_thoughts_renderer(tokenizer):
    if _patch_reasoning_chat_template(tokenizer):
        return "chat_template", None
    return "plain_text", None


def _render_messages_as_text(messages, tokenizer):
    rendered = []
    for message in messages:
        role = message["role"]
        content = message.get("content", "")
        if role == "assistant":
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                content = f"<think>{reasoning}</think>{content}"
        rendered.append(f"{role.capitalize()}: {content}")

    eos = getattr(tokenizer, "eos_token", None)
    text = "\n\n".join(rendered)
    return text + (eos or "")


def _render_open_thoughts_messages(messages, tokenizer, renderer):
    renderer_name, _ = renderer
    if renderer_name == "chat_template":
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return _render_messages_as_text(messages, tokenizer)


def open_thoughts(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
                  shuffle_seed=1234, seed=42, open_thoughts_max_samples=10_000, **_kw):
    rank, world_size = _get_dist_info()

    renderer = _open_thoughts_renderer(tokenizer)

    total_needed = train_samples + val_samples + gpt_samples

    if rank == 0 or world_size <= 1:
        ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
        ds = ds.shuffle(seed=seed).select(range(open_thoughts_max_samples))

        def preprocess(example):
            messages = [{
                "role": "system",
                "content": (
                    "You are Kimi, an AI assistant created by Moonshot AI."
                ),
            }]
            for msg in example["conversations"]:
                role = msg["from"]
                if role == "user":
                    messages.append({"role": "user", "content": msg["value"]})
                else:
                    thought, solution = split_thought_solution(msg["value"])
                    messages.append({"role": "assistant", "content": solution, "reasoning_content": thought})

            return {
                "text": _render_open_thoughts_messages(messages, tokenizer, renderer)
            }

        print(f"Preprocessing {len(ds)} OpenThoughts samples (chat template)...", flush=True)
        ds = ds.map(preprocess, num_proc=min(8, os.cpu_count() or 1))
        print(f"Tokenizing into {total_needed} chunks of length {max_length}...", flush=True)
        all_chunks = make_concat_chunks(ds, tokenizer, max_length, total_needed, seed=shuffle_seed)
        print(f"Broadcasting {len(all_chunks)} chunks to {world_size} ranks...", flush=True)
    else:
        all_chunks = [torch.zeros(max_length, dtype=torch.long)]

    if world_size > 1:
        all_chunks = _broadcast_chunks(all_chunks, rank, world_size)
    if rank == 0:
        print("Dataset ready.", flush=True)

    train_tokens = all_chunks[:train_samples]
    val_tokens = all_chunks[train_samples:train_samples+val_samples]
    gpt_tokens = all_chunks[train_samples+val_samples:]

    train_split = _maybe_shard(train_tokens)
    val_split = _maybe_shard(val_tokens)
    gpt_split = _maybe_shard(gpt_tokens)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(gpt_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, gpt_loader

def create_dataloader(dataset_name, tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length, **kwargs):
    return globals()[dataset_name](tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length, **kwargs)

def _allocate_mixed_counts(total_needed, source_weights):
    if total_needed < 0:
        raise ValueError(f"total_needed must be non-negative, got {total_needed}")
    if len(source_weights) != 3:
        raise ValueError(
            "mixed_source_weights must contain exactly 3 values: "
            "[LLM_compression, OpenThoughts, FineWeb-Edu]"
        )

    weights = [float(weight) for weight in source_weights]
    if any(weight < 0 for weight in weights):
        raise ValueError(f"mixed_source_weights must be non-negative, got {source_weights}")

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError(f"mixed_source_weights must sum to a positive value, got {source_weights}")

    raw_counts = [total_needed * weight / weight_sum for weight in weights]
    counts = [int(count) for count in raw_counts]
    remainder = total_needed - sum(counts)
    by_fraction = sorted(
        range(len(raw_counts)),
        key=lambda idx: raw_counts[idx] - counts[idx],
        reverse=True,
    )
    for idx in by_fraction[:remainder]:
        counts[idx] += 1
    return counts


def mixed(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
          shuffle_seed=1234, seed=42, open_thoughts_max_samples=10_000,
          mixed_source_weights=(0.1, 0.45, 0.45), **_kw):
    rank, world_size = _get_dist_info()
    total_needed = train_samples + val_samples + gpt_samples
    llm_c_count, ot_count, fw_count = _allocate_mixed_counts(total_needed, mixed_source_weights)

    if rank == 0 or world_size <= 1:
        print(
            "Building mixed dataset chunks: "
            f"{llm_c_count} LLM_compression, "
            f"{ot_count} OpenThoughts, "
            f"{fw_count} FineWeb-Edu "
            f"(total={total_needed})",
            flush=True,
        )

        if llm_c_count > 0:
            ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
            ds = ds.shuffle(seed=seed)
            llm_c_tokens = make_message_or_text_chunks(ds, tokenizer, max_length, llm_c_count, seed=shuffle_seed)
        else:
            llm_c_tokens = []
        print(
            f"LLM_compression chunks: requested={llm_c_count}, produced={len(llm_c_tokens)}",
            flush=True,
        )

        ot_requested = ot_count + max(0, llm_c_count - len(llm_c_tokens))
        if ot_requested > 0:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
            ds = ds.shuffle(seed=seed).select(range(min(open_thoughts_max_samples, len(ds))))
            renderer = _open_thoughts_renderer(tokenizer)

            def preprocess(example):
                messages = []
                for msg in example["conversations"]:
                    role = msg["from"]
                    if role == "user":
                        messages.append({"role": "user", "content": msg["value"]})
                    else:
                        thought, solution = split_thought_solution(msg["value"])
                        messages.append({
                            "role": "assistant",
                            "content": solution,
                            "reasoning_content": thought,
                        })
                return {
                    "text": _render_open_thoughts_messages(messages, tokenizer, renderer)
                }

            ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=min(8, os.cpu_count() or 1))
            ot_tokens = make_concat_chunks(ds, tokenizer, max_length, ot_requested, seed=shuffle_seed)
        else:
            ot_tokens = []
        print(
            f"OpenThoughts chunks: requested={ot_requested}, produced={len(ot_tokens)}",
            flush=True,
        )

        fw_requested = total_needed - len(llm_c_tokens) - len(ot_tokens)
        if fw_requested > 0:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split='train', streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=30000)
            fw_tokens = make_concat_chunks(ds, tokenizer, max_length, fw_requested, seed=shuffle_seed)
        else:
            fw_tokens = []
        print(
            f"FineWeb-Edu chunks: requested={fw_requested}, produced={len(fw_tokens)}",
            flush=True,
        )

        all_chunks = llm_c_tokens + ot_tokens + fw_tokens
        if len(all_chunks) < total_needed:
            raise ValueError(
                f"Mixed dataset produced only {len(all_chunks)} chunks, "
                f"but {total_needed} are required."
            )
        random.Random(shuffle_seed).shuffle(all_chunks)
        all_chunks = all_chunks[:total_needed]
        print(f"Broadcasting {len(all_chunks)} mixed chunks to {world_size} ranks...", flush=True)
    else:
        all_chunks = [torch.zeros(max_length, dtype=torch.long) for _ in range(total_needed)]
    if world_size > 1:
        all_chunks = _broadcast_chunks(all_chunks, rank, world_size)
    if rank == 0:
        print("Mixed dataset ready.", flush=True)
    train_tokens = all_chunks[:train_samples]
    val_tokens = all_chunks[train_samples:train_samples + val_samples]
    gpt_tokens = all_chunks[train_samples + val_samples:total_needed]
    train_split = _maybe_shard(train_tokens)
    val_split = _maybe_shard(val_tokens)
    gpt_split = _maybe_shard(gpt_tokens)
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(gpt_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, gpt_loader


NEMOTRON_DATASET = "nvidia/Nemotron-Pretraining-Dataset-sample"


def _iter_nemotron_texts(seed):
    configs = get_dataset_config_names(NEMOTRON_DATASET)
    for cfg in configs:
        ds = load_dataset(NEMOTRON_DATASET, cfg, split="train")
        ds = ds.shuffle(seed=seed)
        for ex in ds:
            text = ex.get("text")
            if text:
                yield {"text": text}


def nemotron_mix(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
                 shuffle_seed=1234, seed=42, open_thoughts_max_samples=10_000, **_kw):
    rank, world_size = _get_dist_info()
    total_needed = train_samples + val_samples + gpt_samples

    if rank == 0 or world_size <= 1:
        print(
            f"Building nemotron_mix chunks (total={total_needed}): "
            "exhaust Nemotron -> exhaust LLM_compression -> fill from OpenThoughts -> backstop FineWeb-Edu",
            flush=True,
        )

        nem_data = _iter_nemotron_texts(seed=seed)
        nem_tokens = make_concat_chunks(nem_data, tokenizer, max_length, total_needed, seed=shuffle_seed)
        print(
            f"Nemotron chunks: requested={total_needed}, produced={len(nem_tokens)}",
            flush=True,
        )

        llm_c_requested = total_needed - len(nem_tokens)
        if llm_c_requested > 0:
            ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
            ds = ds.shuffle(seed=seed)
            llm_c_tokens = make_message_or_text_chunks(ds, tokenizer, max_length, llm_c_requested, seed=shuffle_seed)
        else:
            llm_c_tokens = []
        print(
            f"LLM_compression chunks: requested={llm_c_requested}, produced={len(llm_c_tokens)}",
            flush=True,
        )

        ot_requested = total_needed - len(nem_tokens) - len(llm_c_tokens)
        if ot_requested > 0:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
            ds = ds.shuffle(seed=seed).select(range(min(open_thoughts_max_samples, len(ds))))
            renderer = _open_thoughts_renderer(tokenizer)

            def preprocess(example):
                messages = []
                for msg in example["conversations"]:
                    role = msg["from"]
                    if role == "user":
                        messages.append({"role": "user", "content": msg["value"]})
                    else:
                        thought, solution = split_thought_solution(msg["value"])
                        messages.append({
                            "role": "assistant",
                            "content": solution,
                            "reasoning_content": thought,
                        })
                return {
                    "text": _render_open_thoughts_messages(messages, tokenizer, renderer)
                }

            ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=min(8, os.cpu_count() or 1))
            ot_tokens = make_concat_chunks(ds, tokenizer, max_length, ot_requested, seed=shuffle_seed)
        else:
            ot_tokens = []
        print(
            f"OpenThoughts chunks: requested={ot_requested}, produced={len(ot_tokens)}",
            flush=True,
        )

        fw_requested = total_needed - len(nem_tokens) - len(llm_c_tokens) - len(ot_tokens)
        if fw_requested > 0:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split='train', streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=30000)
            fw_tokens = make_concat_chunks(ds, tokenizer, max_length, fw_requested, seed=shuffle_seed)
        else:
            fw_tokens = []
        print(
            f"FineWeb-Edu chunks: requested={fw_requested}, produced={len(fw_tokens)}",
            flush=True,
        )

        all_chunks = nem_tokens + llm_c_tokens + ot_tokens + fw_tokens
        if len(all_chunks) < total_needed:
            raise ValueError(
                f"nemotron_mix produced only {len(all_chunks)} chunks, "
                f"but {total_needed} are required."
            )
        random.Random(shuffle_seed).shuffle(all_chunks)
        all_chunks = all_chunks[:total_needed]
        print(f"Broadcasting {len(all_chunks)} nemotron_mix chunks to {world_size} ranks...", flush=True)
    else:
        all_chunks = [torch.zeros(max_length, dtype=torch.long) for _ in range(total_needed)]

    if world_size > 1:
        all_chunks = _broadcast_chunks(all_chunks, rank, world_size)
    if rank == 0:
        print("nemotron_mix dataset ready.", flush=True)

    train_tokens = all_chunks[:train_samples]
    val_tokens = all_chunks[train_samples:train_samples + val_samples]
    gpt_tokens = all_chunks[train_samples + val_samples:total_needed]
    train_split = _maybe_shard(train_tokens)
    val_split = _maybe_shard(val_tokens)
    gpt_split = _maybe_shard(gpt_tokens)
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(gpt_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, gpt_loader
