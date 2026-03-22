# python sample_descriptions.py --name Comp_v6_KLD01 --dataset_name t2m --descriptions --descriptions_dir descriptions --repeat_times 1 --gpu_id 0


import argparse
import random
import re
from pathlib import Path
from os.path import join as pjoin

import numpy as np
import spacy
import torch
import torch.nn.functional as F

import utils.paramUtil as paramUtil
from networks.modules import (
    AttLayer,
    MotionLenEstimatorBiGRU,
    MovementConvDecoder,
    MovementConvEncoder,
    TextDecoder,
    TextEncoderBiGRU,
    TextVAEDecoder,
)
from networks.trainers import CompTrainerV6
from scripts.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.plot_script import plot_3d_motion
from utils.utils import motion_temporal_filter
from utils.word_vectorizer import WordVectorizer


def safe_stem(s: str, max_len: int = 120) -> str:
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")
    return s or "caption"


def indices_from_batch(shard_index: int, n: int, shard_size: int):
    if shard_index is None or shard_index < 0 or shard_size is None:
        return set(range(n))
    lo = shard_index * shard_size
    hi = min((shard_index + 1) * shard_size - 1, n - 1)
    if lo > hi:
        return set()
    return set(range(lo, hi + 1))


def apply_index_filter(pending, n_total: int, shard_size: int, shard_index):
    inc_set = indices_from_batch(shard_index, n_total, shard_size)
    return [i for i in pending if i in inc_set]


def load_descriptions_dir(desc_dir: Path):
    """
    Reads <desc_dir>/*.txt in sorted file order.
    Returns:
      prompt_list: list[str] of all non-empty lines
      group_of_prompt: list[str] of matching file stems
    """
    prompt_list = []
    group_of_prompt = []

    for fpath in sorted(desc_dir.glob("*.txt")):
        group = fpath.stem
        for line in fpath.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            prompt_list.append(line)
            group_of_prompt.append(group)

    return prompt_list, group_of_prompt


def load_text_path(text_path: Path):
    """
    Supports legacy text file input:
      prompt
      prompt#64
    """
    prompt_list = []
    group_of_prompt = []
    length_list = []
    has_inline_lengths = True

    for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("#")]
        prompt_list.append(parts[0])
        group_of_prompt.append("text_path")
        if len(parts) > 1 and parts[-1].isdigit():
            length_list.append(int(parts[-1]))
        else:
            has_inline_lengths = False

    if not has_inline_lengths:
        length_list = []

    return prompt_list, group_of_prompt, length_list


def process_text(sentence: str, nlp):
    sentence = sentence.replace("-", "")
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ in ("NOUN", "VERB")) and (word != "left"):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def encode_caption(caption: str, nlp, w_vectorizer: WordVectorizer, max_text_len: int):
    word_list, pos_list = process_text(caption, nlp)
    tokens = [f"{word_list[i]}/{pos_list[i]}" for i in range(len(word_list))]

    if len(tokens) < max_text_len:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    else:
        tokens = tokens[:max_text_len]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)

    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        word_emb, pos_oh = w_vectorizer[token]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])

    return {
        "caption": caption,
        "word_embeddings": np.concatenate(word_embeddings, axis=0).astype(np.float32),
        "pos_one_hots": np.concatenate(pos_one_hots, axis=0).astype(np.float32),
        "sent_len": int(sent_len),
    }


def make_text_batch(encoded_items):
    """
    pack_padded_sequence requires sorted lengths.
    Returns sorted tensors plus an inverse permutation.
    """
    order = sorted(range(len(encoded_items)), key=lambda i: encoded_items[i]["sent_len"], reverse=True)
    inv_order = np.argsort(order)

    word_emb = torch.from_numpy(np.stack([encoded_items[i]["word_embeddings"] for i in order], axis=0))
    pos_ohot = torch.from_numpy(np.stack([encoded_items[i]["pos_one_hots"] for i in order], axis=0))
    cap_lens = torch.LongTensor([encoded_items[i]["sent_len"] for i in order])
    return word_emb, pos_ohot, cap_lens, order, inv_order


def sample_token_length(prob_row: torch.Tensor, min_token_len: int):
    sampled = min_token_len
    for _ in range(3):
        sampled = int(torch.multinomial(prob_row, 1, replacement=True).item())
        if sampled >= min_token_len:
            return sampled
    return max(min_token_len, sampled)


def build_models(opt):
    if opt.text_enc_mod != "bigru":
        raise ValueError(f"Unsupported text encoder mode: {opt.text_enc_mod}")

    text_encoder = TextEncoderBiGRU(
        word_size=opt.dim_word,
        pos_size=opt.dim_pos_ohot,
        hidden_size=opt.dim_text_hidden,
        device=opt.device,
    )
    text_size = opt.dim_text_hidden * 2

    seq_prior = TextDecoder(
        text_size=text_size,
        input_size=opt.dim_att_vec + opt.dim_movement_latent,
        output_size=opt.dim_z,
        hidden_size=opt.dim_pri_hidden,
        n_layers=opt.n_layers_pri,
    )

    seq_decoder = TextVAEDecoder(
        text_size=text_size,
        input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
        output_size=opt.dim_movement_latent,
        hidden_size=opt.dim_dec_hidden,
        n_layers=opt.n_layers_dec,
    )

    att_layer = AttLayer(
        query_dim=opt.dim_pos_hidden,
        key_dim=text_size,
        value_dim=opt.dim_att_vec,
    )

    movement_enc = MovementConvEncoder(
        opt.dim_pose - 4,
        opt.dim_movement_enc_hidden,
        opt.dim_movement_latent,
    )
    movement_dec = MovementConvDecoder(
        opt.dim_movement_latent,
        opt.dim_movement_dec_hidden,
        opt.dim_pose,
    )

    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec


def repeat_done(sample_dir: Path, repeat_id: int, require_mp4: bool):
    tensor_ok = (sample_dir / f"motion_repeat{repeat_id:02d}.pt").exists()
    mp4_ok = (sample_dir / f"motion_repeat{repeat_id:02d}.mp4").exists()
    return tensor_ok and (mp4_ok or not require_mp4)


def load_prompts(args):
    if args.descriptions:
        prompt_list, group_of_prompt = load_descriptions_dir(Path(args.descriptions_dir))
        if not prompt_list:
            raise RuntimeError(f"No descriptions found under {args.descriptions_dir}/*.txt")
        explicit_lengths = []
    elif args.text_prompt:
        prompt_list = [args.text_prompt]
        group_of_prompt = ["single"]
        explicit_lengths = []
    elif args.text_path:
        prompt_list, group_of_prompt, explicit_lengths = load_text_path(Path(args.text_path))
        if not prompt_list:
            raise RuntimeError(f"No prompts found in {args.text_path}")
    else:
        raise RuntimeError("Provide --descriptions, --text_prompt, or --text_path.")

    return prompt_list, group_of_prompt, explicit_lengths


def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "opt.txt")
    opt = get_opt(opt_path, device)
    opt.device = device
    opt.which_epoch = args.which_epoch

    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))

    if opt.dataset_name == "t2m":
        kinematic_chain = paramUtil.t2m_kinematic_chain
        min_token_len = 10
        default_fps = 20.0
        default_radius = 4.0
    elif opt.dataset_name == "kit":
        kinematic_chain = paramUtil.kit_kinematic_chain
        min_token_len = 6
        default_fps = 12.5
        default_radius = 240 * 8
    else:
        raise KeyError(f"Unsupported dataset: {opt.dataset_name}")

    fps = default_fps if args.fps is None else float(args.fps)
    radius = default_radius if args.radius is None else float(args.radius)

    text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec = build_models(opt)
    trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
    epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + ".tar"))
    print(f"Loaded model: epoch={epoch:03d} schedule_len={schedule_len:03d}")
    trainer.eval_mode()
    trainer.to(opt.device)

    estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)
    est_ckpt = torch.load(
        pjoin(opt.checkpoints_dir, opt.dataset_name, "length_est_bigru", "model", "latest.tar"),
        map_location=opt.device,
    )
    estimator.load_state_dict(est_ckpt["estimator"])
    estimator.to(opt.device)
    estimator.eval()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required. Install it with: "
            "python -m spacy download en_core_web_sm"
        ) from e

    w_vectorizer = WordVectorizer("./glove", "our_vab")

    prompt_list, group_of_prompt, explicit_lengths = load_prompts(args)
    encoded_prompts = [encode_caption(c, nlp, w_vectorizer, opt.max_text_len) for c in prompt_list]

    if args.motion_length > 0:
        explicit_frame_lengths = [int(args.motion_length)] * len(prompt_list)
        estimate_lengths = False
    elif explicit_lengths:
        explicit_frame_lengths = [int(x) for x in explicit_lengths]
        estimate_lengths = False
    else:
        explicit_frame_lengths = []
        estimate_lengths = True

    if not estimate_lengths:
        frame_lengths = []
        for length in explicit_frame_lengths:
            adj = max(opt.unit_length, (int(length) // opt.unit_length) * opt.unit_length)
            frame_lengths.append(adj)
    else:
        print("No explicit motion lengths supplied; sampling lengths with the trained length estimator.")
        frame_lengths = [None] * len(prompt_list)
        bs = max(1, int(args.text_batch_size))
        with torch.no_grad():
            for start in range(0, len(encoded_prompts), bs):
                batch_items = encoded_prompts[start:start + bs]
                word_emb, pos_ohot, cap_lens, order, inv_order = make_text_batch(batch_items)
                word_emb = word_emb.to(opt.device).float()
                pos_ohot = pos_ohot.to(opt.device).float()

                pred_dis = estimator(word_emb, pos_ohot, cap_lens)
                probs = F.softmax(pred_dis, dim=-1)

                sampled_token_lens_sorted = [
                    sample_token_length(probs[row], min_token_len=min_token_len)
                    for row in range(probs.shape[0])
                ]
                sampled_token_lens = [sampled_token_lens_sorted[j] for j in inv_order]

                for local_idx, token_len in enumerate(sampled_token_lens):
                    frame_lengths[start + local_idx] = int(token_len) * opt.unit_length

                if opt.device.type == "cuda":
                    torch.cuda.empty_cache()

    result_root = Path(args.result_root) / args.name
    result_root.mkdir(parents=True, exist_ok=True)

    candidate_indices = []
    for i in range(len(prompt_list)):
        sample_dir = result_root / group_of_prompt[i] / str(i)
        incomplete = any(
            not repeat_done(sample_dir, r, require_mp4=not args.no_mp4)
            for r in range(args.repeat_times)
        )
        if incomplete:
            candidate_indices.append(i)

    pending_indices = apply_index_filter(
        pending=candidate_indices,
        n_total=len(prompt_list),
        shard_size=args.shard_size,
        shard_index=args.shard_index,
    )

    if args.shard_index is not None:
        print(
            f"Index filter applied (shard_index={args.shard_index}, shard_size={args.shard_size}). "
            f"Now considering {len(pending_indices)} prompts out of {len(prompt_list)} total."
        )

    if not pending_indices:
        print(f"All requested outputs already exist under {result_root}. Nothing to do.")
        return

    print(f"Generating {len(pending_indices)} prompts under {result_root}")
    print(f"First few global indices: {pending_indices[:10]}")

    with torch.no_grad():
        for r in range(args.repeat_times):
            print(f"--> Repeat {r}")
            for orig_idx in pending_indices:
                caption = prompt_list[orig_idx]
                group = group_of_prompt[orig_idx]
                sample_dir = result_root / group / str(orig_idx)
                sample_dir.mkdir(parents=True, exist_ok=True)

                if repeat_done(sample_dir, r, require_mp4=not args.no_mp4):
                    continue

                seed_for_output = int(args.seed + 1000003 * orig_idx + 9176 * r)
                random.seed(seed_for_output)
                np.random.seed(seed_for_output % (2**32 - 1))
                torch.manual_seed(seed_for_output)
                if opt.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed_for_output)
                enc = encoded_prompts[orig_idx]
                word_emb = torch.from_numpy(enc["word_embeddings"][None, ...]).to(opt.device).float()
                pos_ohot = torch.from_numpy(enc["pos_one_hots"][None, ...]).to(opt.device).float()
                cap_lens = torch.LongTensor([enc["sent_len"]])

                m_len = int(frame_lengths[orig_idx])
                m_lens = torch.LongTensor([m_len]).to(opt.device)
                mov_len = int(m_len // opt.unit_length)

                print(f"    ----> {group}/{orig_idx}: {caption}  len={m_len}")
                pred_motions, _, att_wgts = trainer.generate(
                    word_emb,
                    pos_ohot,
                    cap_lens,
                    m_lens,
                    mov_len,
                    opt.dim_pose,
                )

                pred_motion = pred_motions[0, :m_len].detach().cpu().numpy()
                ric_motion = pred_motion * std + mean
                joints = recover_from_ric(torch.from_numpy(ric_motion).float(), opt.joints_num).numpy()

                if args.smooth_sigma > 0:
                    joints = motion_temporal_filter(joints, sigma=args.smooth_sigma)

                print(
                    "       joint stats:",
                    "min", np.nanmin(joints),
                    "max", np.nanmax(joints),
                    "nan?", np.isnan(joints).any(),
                    "inf?", np.isinf(joints).any(),
                )

                (sample_dir / "prompt.txt").write_text(caption + "\n", encoding="utf-8")
                torch.save(torch.from_numpy(joints.astype(np.float32)), sample_dir / f"motion_repeat{r:02d}.pt")
                np.save(sample_dir / f"motion_ric_repeat{r:02d}.npy", ric_motion.astype(np.float32))
                np.save(sample_dir / f"att_weights_repeat{r:02d}.npy", att_wgts[0].detach().cpu().numpy())

                if not args.no_mp4:
                    plot_3d_motion(
                        str(sample_dir / f"motion_repeat{r:02d}.mp4"),
                        kinematic_chain,
                        joints,
                        title=caption,
                        fps=fps,
                        radius=radius,
                    )

                if opt.device.type == "cuda":
                    torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", required=True, help="Run name under checkpoints/<dataset>/<name>/")
    parser.add_argument("--dataset_name", type=str, default="t2m")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--which_epoch", type=str, default="latest")
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--text_prompt", type=str, default="")
    parser.add_argument("--text_path", type=str, default="")
    parser.add_argument("--descriptions", action="store_true", help="Read prompts from descriptions_dir/*.txt")
    parser.add_argument("--descriptions_dir", type=str, default="descriptions")

    parser.add_argument("--motion_length", type=int, default=0, help="Optional explicit motion length in frames.")
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--text_batch_size", type=int, default=64, help="Batch size only for length estimation.")

    parser.add_argument("--result_root", type=str, default="generation")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--no_mp4", action="store_true")

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=None)

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())