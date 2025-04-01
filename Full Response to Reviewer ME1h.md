## Opening

Thank you for your valuable feedback!

## Exploring LoRA-FA Performance Across Different Datasets and Identifying Patterns

Based on our experimental results, we acknowledge that LoRA-FA does not always outperform alternative approaches. Although LoRA-FA serves as an efficient fine-tuning method designed to closely approximate the gradients of Full-FT while significantly reducing activation memory overhead through gradient approximation, its performance may vary across different tasks and datasets.

Currently, LoRA-FA belongs to the category of low-rank gradient optimization methods whose explicit goal is to approximate the gradients of full fine-tuning (Full-FT). Within this scope, LoRA-FA primarily competes with LoRA-GA, LoRA-Pro, and DoRA. In contrast, other LoRA-like methods—such as VeRA and QLoRA—focus primarily on reducing trainable parameters or minimizing overall model storage, rather than explicitly approximating Full-FT gradients.

To better illustrate the comparative performance, we conducted additional experiments evaluating Full-FT, LoRA-GA, LoRA-Pro, and LoRA-FA on the MMLU dataset. The results are summarized as follows:

| Method   | MMLU (%) |
| -------- | -------- |
| Full-FT  | 57.8     |
| LoRA-GA  | 55.9     |
| LoRA-Pro | 56.6     |
| LoRA-FA  | 56.2     |

These results demonstrate remarkable consistency among the three gradient approximation methods (LoRA-Pro, LoRA-FA, LoRA-GA), with LoRA-Pro slightly outperforming LoRA-FA, and LoRA-FA in turn marginally outperforming LoRA-GA. However, it is noteworthy that Full-FT's performance advantage is quite limited in this scenario. A plausible explanation is that when the base model already possesses substantial capability (e.g., Llama2-7B achieves a 45.9% accuracy on 5-shot MMLU without fine-tuning), even advanced low-rank gradient optimization methods have limited room for further improvement. In such situations, LoRA-FA and related methods reach performance levels similar to Full-FT, and thus the efficiency benefits of LoRA-FA (such as reduced activation memory) become more pronounced relative to its marginal performance gap.

| Initialization method | GSM8K |
| --- | --- |
| kaiming uniform | 57.3 |
| gaussian | 53.7 |
| PiSSA | 10.6 |
| OLoRA | 3.1 |
| EVA | 55.3 |

Regarding other methods, particularly those based on decomposition of the pretrained weight matrix $W$ (e.g., PiSSA, DoRA, OLoRA), we observe fundamental limitations. Specifically, successful low-rank fine-tuning initialization requires the difference between the base weight and the initialized low-rank approximation to be sufficiently small, i.e.,  

$$\|W - A_0 B_0\| < \varepsilon.$$  

Only certain initialization methods, such as Kaiming uniform, Gaussian, and EVA meet this condition effectively. In contrast, methods like PiSSA and DoRA, which rely on singular-value decomposition (SVD), inherently introduce substantial approximation errors determined by the product of the smallest $m - r$ singular values of $W$. As a result, these decomposition-based methods frequently encounter convergence difficulties or suboptimal performance.

Additionally, we observe that methods employing dynamic adjustments of learning rate or rank (e.g., LoRA+ and AdaLoRA) can indeed yield improvements in performance. However, their advantages are typically smaller compared to those obtained from explicit gradient approximation methods such as LoRA-FA, LoRA-GA, and LoRA-Pro.

## Closing Remarks

We have updated the manuscript to include these additional experimental results and analyses. Detailed theoretical insights are provided in references [Theoretical analysis of generalization error bound.md] and [Proof of Theorem 2.1.pdf]. We are actively preparing LoRA-FA for integration with PEFT, allowing the community to benefit from its efficiency soon. Despite its advantages, LoRA-FA remains inherently constrained by its low-rank approximation nature and potential catastrophic forgetting. Addressing these limitations, particularly approximation accuracy and forgetting phenomena, represents a promising direction for future work.

Thank you again for your constructive feedback, greatly enhancing the clarity and impact of our research!
