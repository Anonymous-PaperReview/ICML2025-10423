## Opening

Thank you very much for your insightful feedback and valuable suggestions! Please find below our detailed responses to your comments.

> My only concern is that given there are various LoRA-based fine-tuning methods, such as SVD-fine-tune [1] and VeRA [2], that push the number of trainable parameters to the limit, it is unclear how this method compares with those approaches utilizing even fewer parameters.

Thank you for highlighting this important point. Indeed, there are several recent methods that employ aggressive strategies to significantly reduce the number of trainable parameters, notably Tied-LoRA and VeRA. Regarding SVD-based fine-tuning methods such as SVFT, we note that their total number of trainable parameters remains relatively substantial compared to methods like VeRA. Specifically, SVFT involves decomposing the original parameter matrix $W$ via singular value decomposition (SVD), and the resulting trainable portions retain dimensions similar to $W$ itself. For instance, the best-performing SVFT configuration reported in [1] requires approximately 6.35 million trainable parameters, which is nearly ten times larger than that of VeRA. Thus, in our subsequent analysis, we focus primarily on VeRA and Tied-LoRA, as these methods represent more aggressive parameter reduction strategies.

After carefully reviewing Tied-LoRA, we observed significant conceptual similarities with VeRA. Both approaches globally share the low-rank matrices $A$ and $B$ across the entire model, with fine-tuning limited mainly to vectors $u$ and $v$. Due to the unavailability of an official implementation for Tied-LoRA, we considered it as a special case or variant of VeRA. To empirically assess the relative performance of these methods, we selected Tied-LoRA's optimal configuration (denoted TL6), fine-tuned a LLaMA2-7B model on the Metamath40k dataset, and measured performance on the GSM8K benchmark. The results are summarized below:

| Method                           | GSM8K |
| -------------------------------- | ----- |
| LoRA-FA                          | 57.3  |
| Tied-LoRA (TL6, train A,B,u,v)   | 43.3  |
| VeRA (train u,v only)            | 42.9  |

As the results indicate, both VeRA and Tied-LoRA exhibit notably lower performance compared to LoRA-FA. This observed performance gap is reasonable and aligns with expectations, given that VeRA and Tied-LoRA deliberately pursue aggressive reductions in trainable parameters at the expense of fine-tuning accuracy. Consequently, these methods differ fundamentally from LoRA-FA, whose explicit goal is to closely approximate the performance of full fine-tuning (Full-FT) via precise gradient approximation. Thus, while VeRA and Tied-LoRA provide valuable solutions when parameter efficiency (e.g., rapid task switching, multi-LoRA setups, or parameter storage constraints) is paramount, LoRA-FA primarily targets scenarios requiring performance closer to full fine-tuning, albeit with moderate increases in parameter count.

## Closing Remarks

We have updated the manuscript to include these additional experimental results and analyses. Detailed theoretical insights are provided in references [Theoretical analysis of generalization error bound.md] and [Proof of Theorem 2.1.pdf]. We are actively preparing LoRA-FA for integration with PEFT, allowing the community to benefit from its efficiency soon. Despite its advantages, LoRA-FA remains inherently constrained by its low-rank approximation nature and potential catastrophic forgetting. Addressing these limitations, particularly approximation accuracy and forgetting phenomena, represents a promising direction for future work.

Thank you again for your constructive feedback, greatly enhancing the clarity and impact of our research!

[1] SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors

[2] Tied-Lora: Enhancing parameter efficiency of LoRA with weight tying
