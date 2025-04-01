# lorafa

## For PEFT ref ONLY

1. install llmtoolkit

```bash
git clone https://github.com/AaronZLT/llm-toolkit.git
cd llm-toolkit && pip install .
```

2. run finetune

```bash
cd metamath && ./run.sh
cd mmlu && ./run.sh
```

3. run evaluate

```bash
cd metamath && python eval_gsm8k.py
cd mmlu && python eval_mmlu.py
```

4. benchmark system efficiency

To benchmark, just add --profiler pytorch.
