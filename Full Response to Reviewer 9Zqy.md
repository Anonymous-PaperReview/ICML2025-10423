## Opening

Thank you very much for your valuable feedback!

> LoRA-FA's theoretical proof and the initialization of the fixed weight matrix significantly impact performance.

We first acknowledge that Theorem 2.1, as originally stated, was not sufficiently rigorous. As you correctly pointed out, the equality  

$$\Delta W = A^* B^* = A_0 B'^*$$  

does not strictly hold. Our intended statement was one of approximation rather than strict equality. A more rigorous formulation should indeed be:  

$$\Delta W = A^* B^* \approx A_0 B'^*.$$

To address this, we have enhanced our theoretical analysis to explicitly quantify the approximation error. Specifically, we now provide an analysis to calculate the Expected Value of $\| A^* - A_0 C^* \|_F^2$ When $A^*$ and $A_0$ are under $N(0,1)$ distribution. Please refer to reference [Proof of Theorem 2.1.pdf] for detailed derivations and discussions.

We would like to take this opportunity to further elaborate on LoRA-FA’s theoretical foundations and practical implications. Initially, LoRA-FA was conceived as a predominantly empirical exploration shortly after the introduction of LoRA. We observed a notable asymmetry between matrices A and B; intuitively, matrix A acts primarily as a transmitter, whereas B serves as a feature extractor—analogous, in our view, to how attention mechanisms interact with MLP layers in transformer models. Motivated by this observation, we conducted extensive experiments validating that training matrix B alone could achieve competitive performance compared to standard LoRA (though not fully matching LoRA), with significant memory savings for activations.

However, rigorously proving the effectiveness of LoRA-FA posed considerable theoretical challenges. We gratefully acknowledge the prior work [1], which provided valuable theoretical insights by deriving the generalization error bounds for LoRA-FA and LoRA. Specifically, leveraging fundamental principles from generalization error analysis [2], [1] successfully demonstrated that the generalization error bound of LoRA-FB serves as an upper bound for LoRA-FA. Furthermore, it established that the generalization error bound of LoRA-FA is at most a factor of $\sqrt{2}$ smaller than that of LoRA. This result implies a practical guideline: one can theoretically approximate LoRA's performance by doubling the rank $r$ in LoRA-FA. The detailed proof is provided in reference [Theoretical analysis of generalization error bound.md].

Given these considerations, quantifying the performance gap between LoRA-FA and LoRA (or full fine-tuning) becomes crucial for validating LoRA-FA's effectiveness. However, the generalization error bounds derived in [1] serve primarily as theoretical lower bounds, which are not directly indicative of whether LoRA-FA can match LoRA’s optimal empirical performance. To bridge this gap, we pursued an alternative approach to analyze whether LoRA-FA could directly converge to a solution comparable to or identical to that of LoRA, as presented in our proof in section A.1 of the manuscript.

> In Fig. 2, activation memory occupies most of the memory usage, which seems counterintuitive. The authors should provide further explanation and discussion regarding memory usage.

To compute gradients for a linear layer, it is necessary to store activations (inputs) from forward propagation. Concretely, when computing gradients for matrix A, we require its input activation X, which coincides exactly with the input to the original weight matrix W. Similarly, the gradient computation for matrix B requires its input activation, which is not X but rather the product XA. Thus, the matrix A input activation X has dimensions (batch size × sequence length × hidden dimension), while the input activation XA for matrix B has significantly smaller dimensions (batch size × sequence length × rank r). Typically, the chosen rank r is far smaller than the hidden dimension (for example, r=16 versus hidden dimension=4096 in Llama3-8B). Consequently, the activation memory requirement for training matrix B is substantially lower, at least 256 times smaller in the mentioned example, compared to training matrix A.

The above case specifically illustrates activations associated with matrices A and B. More broadly, activation values are generated and stored extensively throughout the model, including between layers, between attention mechanisms, and within MLP modules. As shown in our manuscript's Appendix, we provide detailed breakdowns of activation memory requirements for each model module.

Furthermore, in low-rank fine-tuning (e.g., LoRA), trainable parameters are limited exclusively to adapters, greatly reducing gradient and optimizer state memory usage. Consequently, activations and model parameters become the dominant memory consumers. As batch size and sequence length grow, activations increasingly dominate memory usage, eventually surpassing even model parameters and becoming the primary memory bottleneck.

> Fig. 3 is unclear. It is difficult to discern which method each "OOM" label corresponds to.

Thank you for highlighting this issue. We have improved the clarity of Fig. 3 by explicitly labeling each method along the x-axis of the revised bar chart. This correction will be incorporated in the next revision of the manuscript to ensure clarity and readability.

## Closing Remarks

We have updated the manuscript to include these additional experimental results and analyses. Detailed theoretical insights are provided in references [Theoretical analysis of generalization error bound.md] and [Proof of Theorem 2.1.pdf]. We are actively preparing LoRA-FA for integration with PEFT, allowing the community to benefit from its efficiency soon. Despite its advantages, LoRA-FA remains inherently constrained by its low-rank approximation nature and potential catastrophic forgetting. Addressing these limitations, particularly approximation accuracy and forgetting phenomena, represents a promising direction for future work.

Thank you again for your constructive feedback, greatly enhancing the clarity and impact of our research!

[1] Asymmetry in Low-Rank Adapters of Foundation Models

[2] Information-theoretic analysis of generalization capability of learning algorithms
