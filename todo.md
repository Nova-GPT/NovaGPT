
(B) #RND – Arush: find a better way to validate current one, which only gives a number; training parameter also seems dicey. {cr:2025-10-20}

(B) #RND – Arush: validate current loss/accuracy method more rigorously {cr:2025-10-20} +Validation  
- [ ] Check if our current implementation is **individual** or **semi-batch** (i.e., are we updating after each run or after gradient accumulation?)

(B) #RND Explore concept of **Progressive Context Increment** {cr:2025-10-20} +TrainingOptimization
  Start with block size 64 → 128 → 256 → ... for more efficient training.

(B) #RND – Explore the **paper reducing training time from quadratic to linear** {cr:2025-10-20} +Efficiency

(B) #RND – Devansh: explore **Multi-Head Latent Attention (MLA)** in DeepSeek-V2 & R1 for relevance. {cr:2025-10-20} +Attention

(C) #RND – Explore various **DeepSeek innovations**  🔗 https://medium.com/@jannadikhemais/the-engineering-innovations-behind-deepseek-how-a-chinese-startup-redefined-ai-efficiency-90ea30788829   {cr:2025-10-20} +DeepSeek {c}
  (C) Arush – Read about **RL in DeepSeek** {cr:2025-10-20} +Reading

(C) #RND – Understand **tricks used in SmolLM 1 → 3** {cr:2025-10-20} +SmolLM 

(C) #RND – Analyze trained weights: determine which are effective vs redundant {cr:2025-10-20} +ModelAnalysis

(A) Implement **Automatic Mixed Precision (AMP)** for 2–3× speed & 50% less memory {cr:2025-10-20} +Performance {due:2025-10-25} {cm:2025-10-20} {h}


(B) Arush – Create **training process grapher** (validation + test time) {cr:2025-10-20} +Visualization

(C) Pranav – Implement **Mixture of Experts (MoE)** layer {cr:2025-10-20} +Architecture {cm:2025-10-22} {h}

(B) Implement **token frequency analyzer** {cr:2025-10-20} +DataStats  
  Count occurrences of each token in DB 
  Display histogram of vocab coverage

(C) Add **loading bar with percentage** to indicate dataset coverage {cr:2025-10-20} +UX


(A) **LORA like approximating initial step** : instead of directly training a ab layer, first train a2 * 2b layer first then jump upto a * b layer #RND #Arush +ProgressiveTrainingStratagy
(A) Torch.compile implimentation #Engineering #Arush
(B) Impliment Gradient Accumilation #Engineering #Arush
(B) **Experiment with ideal Vocab size** 50000 is an overkill #RND #Arush
(C) **Implimentation of DeepSeek MLA** - gains for all model sizes
(C) **Compare MLA vs WGQA-4 vs WGQA-8** - understand the various tradeoffs
(C) **Impliment Cosine Learning Rate Schedule** - Found this in Karpathy Video

(A) **Implimentation of Flash Attention Kernals(v2)** : will speedup the attention part
(A) **Enable cuDNN / cuBLAS auto-tuning** : not very sure what or how it does it, but it speeds up things
(A) **Gradient Accumilation Implimentation** : this will simulate a larger batch size without increasing the vram : simple to impliment also
(A) **Use torch.utils.checkpoint** : Saves ~30–40% memory at the cost of ~15–20% slower compute