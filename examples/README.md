# SCANNER Training Configurations

These configurations train FLUX.1-dev or Qwen-Image-2512 with TextDART rewards.
Install the [SCANNER environment](../README.md#part-ii-generative-model-rl-scanner)
and [deploy TextDART](../TextDART/train/README.md#2-quick-start-run-the-model)
before launching training.

## Configurations

- `grpo/lora/flux1/textdart_scanner.yaml` — FLUX.1-dev
- `grpo/lora/qwen_image_2512/textdart_scanner.yaml` — Qwen-Image-2512

Both expect a deployed TextDART server; replace the
`<TODO: TEXTDART_SERVER_HOST>` hosts in `rewards` / `eval_rewards` before launching.

Both configurations were tuned for 4 nodes with 8 GPUs per node. Start the
same launcher from the `TextDART_SCANNER` project root on every node:

```bash
bash multinode_examples/grpo/flux1/textdart_scanner_4nodes.sh
```

See [`../multinode_examples/README.md`](../multinode_examples/README.md) for
the required scheduler variables and launch details.
