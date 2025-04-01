## Opening

Thank you very much for your insightful feedback and valuable suggestions! Please find below our detailed responses to your comments.

> Combining low-rank gradient optimization methods with distributed training approaches like ZeRO-1 and ZeRO-2 poses challenges, as these methods flatten gradients and optimizer states. Do the authors have insights or potential solutions for addressing this limitation?

We first summarize our conclusion: When both the inputs and outputs of low-rank gradient optimization methods can be accessed simultaneously, integration with ZeRO methods is naturally feasible, as ZeRO essentially introduces additional communication steps without fundamentally changing the computational nature of gradient updates. Before diving into the detailed discussion, we define computational feasibility by clearly identifying inputs and outputs for these methods. For instance, in LoRA-FA, inputs include the weights of matrices A and B, gradients of B, and optimizer states associated with B; outputs comprise updated gradients and optimizer states of B.

We now provide an in-depth analysis of how low-rank gradient optimization methods could be integrated with ZeRO:

### ZeRO-1

In ZeRO-1, only optimizer states are sharded, while parameters remain fully replicated across all workers at all times. After the backward pass, gradients are synchronized via ReduceScatter. Thus, gradients become ready immediately after this step, enabling methods that rely on accessing both parameters and gradients simultaneously (such as LoRA-FA) to proceed without issue.

### ZeRO-2

ZeRO-2 shards both optimizer states and gradients. However, this scenario does not differ fundamentally from ZeRO-1. Gradients are similarly synchronized via ReduceScatter after the backward pass, rendering gradients immediately available post-communication, just as in ZeRO-1.

From the above analysis, we observe a consistent pattern: computational efficiency significantly increases when inputs and outputs become simultaneously accessible. Current gradient approximation methods, including LoRA-Pro, LoRA-FA and LoRA-GA, restrict their computation within the input-output range. Specifically, LoRA-GA requires only gradients of W to compute updated A, B, and W matrices, while LoRA-Pro and LoRA-FA involves only parameters, gradients, and optimizer states of A and B. These approaches essentially represent in-place updates. Thus, when integrating with ZeRO, low-rank gradient optimization methods need merely identify the communication phase when their required inputs become accessible.

Nevertheless, we acknowledge that explicitly adapting LoRA-FA for ZeRO integration requires additional development efforts. Furthermore, given that LoRA methods inherently possess minimal optimizer states, ZeRO may offer limited memory savings.

> Why were experiments not conducted using different learning rates? Typically, matrix B requires a larger learning rate compared to matrix A due to gradient imbalance. It would be intriguing to explore if LoRA-FA necessitates a larger learning rate than LoRA.

This observation is indeed insightful and aligns closely with the core motivation behind LoRA+. LoRA+ explicitly applies different learning rates (LRs) to matrices A, B, and embeddings to improve convergence. Specifically, the LR of matrix B in LoRA+ is scaled relative to matrix A's LR by a scalar factor (`loraplus_lr_ratio`), with a suggested ratio of 16.

We conducted additional experiments to explore this direction. Setting LoRA's baseline LR to 7e-5, we varied the `loraplus_lr_ratio` within {1, 2, 4, 8, 16, 32} for LoRA-FA. Results from fine-tuning LLaMA2-7B on Metamath40k and evaluating GSM8K scores are as follows:

| loraplus_lr_ratio | GSM8K |
| --- | --- |
| LoRA: 7e-5 | 44.5 |
| LoRA-FA: loraplus_lr_ratio = 1 | 52.1 |
| LoRA-FA: loraplus_lr_ratio = 2 | 55.7 |
| LoRA-FA: loraplus_lr_ratio = 4 | 57.0 |
| LoRA-FA: loraplus_lr_ratio = 8 | 50.4 |
| LoRA-FA: loraplus_lr_ratio = 16 | 49.5 |
| LoRA-FA: loraplus_lr_ratio = 32 | 42.9 |

From these results, we observe improved performance with increasing LR until a peak at `loraplus_lr_ratio = 4`, differing notably from LoRA+'s recommended ratio of 16. This clearly indicates that LoRA-FA indeed benefits from larger learning rates compared to standard LoRA, suggesting an intriguing avenue for future investigation.

> Have the authors considered integrating LoRA-FA with alternative initialization strategies? For instance, leveraging singular-value-based features of pre-trained weights or gradients to initialize matrix A could yield interesting results. What are the authors' thoughts?

Your suggestion is insightful. Given that LoRA-FA's gradient optimization heavily relies on matrices A and B initialization, exploring effective initialization strategies is indeed critical. Recent related studies have investigated initializing A and B by decomposing the base weight W, such as OLoRA (QR decomposition), PiSSA (SVD decomposition on W), and EVA (SVD on activations).

We conducted experiments using these initialization methods alongside Gaussian initialization (A~N(0, 1/r), B=0, the default in LoRA), and compared them against our standard kaiming uniform initialization:

| Initialization method | GSM8K |
| --- | --- |
| kaiming uniform | 57.3 |
| gaussian | 53.7 |
| PiSSA | 10.6 |
| OLoRA | 3.1 |
| EVA | 55.3 |

Results indicate our default kaiming uniform initialization provided the best outcome, followed by EVA and Gaussian. In contrast, PiSSA and OLoRA failed to converge satisfactorily. A plausible explanation is that initializing near zero or with minimal deviation (kaiming, Gaussian, EVA) provides unbiased starting points, whereas PiSSA and OLoRA inherently introduce approximation errors proportional to discarded singular values, hindering convergence.

Consider base weight W and A B in LoRA. The successful training needs $W-A_0B_0<\varepsilon$, and only kaiming uniform, gaussian, EVA holds. The least error of PiSSA and OLoRA is the multiply of top(m-r) smallest singular-values of W.

> Can the authors compare LoRA-FA with Tied-LoRA, which similarly freezes matrices to improve efficiency?

After carefully reviewing Tied-LoRA, we found notable similarities with VeRA, where matrices A and B are globally shared, and only vectors u and v are fine-tuned. Due to the absence of an official Tied-LoRA implementation, we treated it as a special case of VeRA. We selected Tied-LoRA's best configuration (TL6) for fine-tuning LLaMA2-7B on Metamath40k, obtaining the following GSM8K results:

| Method | GSM8K |
| --- | --- |
| LoRA-FA | 57.3 |
| Tied-LoRA (TL6, train A,B,u,v) | 43.3 |
| VeRA (train u,v only) | 42.9 |

The observed performance gap is reasonable, given that VeRA and Tied-LoRA aggressively reduce trainable parameters at the cost of performance. Thus, these methods differ significantly from LoRA-FA's goal of approximating full fine-tuning performance.

> The authors did not explicitly explain why training matrix B requires less activation memory compared to matrix A.

To compute gradients for a linear matrix, we need its corresponding activation inputs. Computing A's gradient requires input X (batch × sequence length × hidden dimension), identical to the original weight W's input. Conversely, B's gradient requires input XA (batch × sequence length × rank r). Given r ≪ hidden dimension (e.g., r=16 vs. hidden dimension=4096 in LLaMA3-8B), training B involves significantly less activation memory, approximately 256 times smaller in this example.

### Comments about LoRA-Pro & LoRA+

Thank you for highlighting these points. We will clarify the references and descriptions of LoRA-Pro in subsequent revisions. Regarding LoRA+, we note its modest experimental performance in our replication, hence our baseline aligns more closely with LoRA-Pro and LoRA-GA. Currently LoRA+ in PEFT will indeed suffer OOM when finetune Llama2-7B at 1024 seq_length and 8 batch size. We are still checking the reason, and we think there is an issue in the init of LoRA+ optimizer.

### Typo Correction

Thank you for pointing out the typo ("[reference needed]"). We will correct this in the next manuscript revision.

## Closing Remarks

We have updated the manuscript to include these additional experimental results and analyses. Detailed theoretical insights are provided in references [Theoretical analysis of generalization error bound.md] and [Proof of Theorem 2.1.pdf]. We are actively preparing LoRA-FA for integration with PEFT, allowing the community to benefit from its efficiency soon. Despite its advantages, LoRA-FA remains inherently constrained by its low-rank approximation nature and potential catastrophic forgetting. Addressing these limitations, particularly approximation accuracy and forgetting phenomena, represents a promising direction for future work.

Thank you again for your constructive feedback, greatly enhancing the clarity and impact of our research!

[1] PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models

[2] OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models

[3] One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation
