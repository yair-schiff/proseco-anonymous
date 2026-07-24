import os

import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def add_gumbel_noise(logits, temperature):


    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:


    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)
    base  = torch.div(total, steps, rounding_mode="floor")
    rem   = total - base * steps


    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)


    cols = torch.arange(steps, device=device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=256,
    gen_length=256,
    block_length=32,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    threshold=None,
    max_corrector_steps_per_loop=4,
    apply_corrector_every_n_steps=2,
    early_eos_stopping=True,
    tokenizer=None,
    disable_pbar=True,
    save_intermediate_outputs=False,
):


    prompt_len = prompt.shape[1]
    batch_size = prompt.shape[0]


    x = torch.full(
        (batch_size, prompt_len + gen_length), mask_id, dtype=torch.long,
    ).to(model.device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        )
    steps_per_block = steps // num_blocks

    predictor_nfe = 0
    corrector_nfe = 0
    total_nfe = 0
    intermediate_outputs = []

    block_pbar = tqdm(range(num_blocks), leave=False, disable=disable_pbar)
    for block_idx in block_pbar:
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length

        block_mask = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)


        active_region_start = prompt_len

        step = 0
        applied_corrector = False

        while True:

            predictor_nfe += 1
            total_nfe += 1

            global_mask = (x == mask_id)
            logits = model(x).logits


            global_mask[:, block_end:] = False
            active_mask = global_mask[:, active_region_start:block_end]


            if (step + 1) % apply_corrector_every_n_steps == 0:
                corrector_step = 0

                if max_corrector_steps_per_loop > 0:
                    corrector_x = x.clone()


                    corrector_x[:, active_region_start:] = torch.argmax(
                        logits, dim=-1,
                    )[:, active_region_start:]

                    corrector_x[:, active_region_start:block_end][~active_mask] = (
                        x[:, active_region_start:block_end][~active_mask]
                    )
                    applied_corrector = True
                else:
                    corrector_x, corrector_logits = None, None

                while corrector_step < max_corrector_steps_per_loop:
                    corrector_step += 1
                    corrector_nfe += 1
                    total_nfe += 1
                    block_pbar.set_postfix(
                        predictor_nfe=predictor_nfe,
                        corrector_nfe=corrector_nfe,
                        total_nfe=total_nfe,
                    )
                    corrector_logits = model(corrector_x).logits[
                        :, active_region_start:block_end
                    ]
                    corrected_tokens = torch.argmax(corrector_logits, dim=-1)

                    if torch.allclose(
                        corrector_x[:, active_region_start:block_end],
                        corrected_tokens,
                    ):
                        break
                    corrector_x[:, active_region_start:block_end] = corrected_tokens


                if max_corrector_steps_per_loop > 0:
                    logits[:, active_region_start:block_end] = corrector_logits
                    active_mask = global_mask[:, active_region_start:block_end]
                    x[:, active_region_start:block_end][~active_mask] = (
                        corrected_tokens[~active_mask]
                    )


            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                global_mask,
                x,
                num_transfer_tokens[:, step] if threshold is None else None,
                threshold,
            )
            x[transfer_index] = x0[transfer_index]
            step += 1

            block_pbar.set_postfix(
                predictor_nfe=predictor_nfe,
                corrector_nfe=corrector_nfe,
                total_nfe=total_nfe,
            )

            if save_intermediate_outputs:
                intermediate_outputs.append({
                    "step": step,
                    "applied_corrector": applied_corrector,
                    "output": x[:, prompt_len:block_end].detach().cpu(),
                })


            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break


        if (
            early_eos_stopping
            and tokenizer is not None
            and (x[:, block_end - 1] == tokenizer.eos_token_id).all()
        ):
            x[x == mask_id] = tokenizer.eos_token_id
            break

    return x, {
        "predictor_nfe": predictor_nfe,
        "corrector_nfe": corrector_nfe,
        "total_nfe": total_nfe,
    }, intermediate_outputs


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens,
    threshold: float = None,
):


    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)


    if remasking == "low_confidence":

        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)


    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(
        torch.finfo(x0_p.dtype).min,
        device=x0_p.device,
        dtype=x0_p.dtype
    )
    confidence = torch.where(mask_index, x0_p, neg_inf)


    if threshold is not None:


        transfer_index = mask_index & (confidence >= threshold)


        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)


        transfer_index = transfer_index | force_mask


        transfer_index = transfer_index & mask_index

        return x0, transfer_index


    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")


    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)


    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape

    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded


    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index

    return x0, transfer_index


def main():
    device = "cuda"
    model_path = os.environ.get("MODEL_PATH")
    if model_path is None:
        raise ValueError("MODEL_PATH must be set.")

    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    prompt = ("A runner travels at 12 kilometers per hour for 4 hours, then at "
              "6 kilometers per hour for 4 hours. What total distance is traveled?")


    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate(
            model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking="low_confidence")

        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()
