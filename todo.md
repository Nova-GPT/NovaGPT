(B) #RND Explore concept of **Progressive Context Increment** {cr:2025-10-20} +TrainingOptimization
  Start with block size 64 → 128 → 256 → ... for more efficient training.

(B) #RND – Explore the **paper reducing training time from quadratic to linear** {cr:2025-10-20} +Efficiency

(C) #RND – Explore various **DeepSeek innovations**  🔗 https://medium.com/@jannadikhemais/the-engineering-innovations-behind-deepseek-how-a-chinese-startup-redefined-ai-efficiency-90ea30788829   {cr:2025-10-20} +DeepSeek {c}
  (C) Arush – Read about **RL in DeepSeek** {cr:2025-10-20} +Reading

(C) #RND – Understand **tricks used in SmolLM 1 → 3** {cr:2025-10-20} +SmolLM 

(C) #RND – Analyze trained weights: determine which are effective vs redundant {cr:2025-10-20} +ModelAnalysis

(B) Implement **token frequency analyzer** {cr:2025-10-20} +DataStats  
  Count occurrences of each token in DB 
  Display histogram of vocab coverage

(A) **LORA like approximating initial step** : instead of directly training a ab layer, first train a2 * 2b layer first then jump upto a * b layer #RND #Arush +ProgressiveTrainingStratagy
(B) **Experiment with ideal Vocab size** 50000 is an overkill #RND #Arush
(C) **Compare MLA vs WGQA-4 vs WGQA-8** - understand the various tradeoffs
(C) **Impliment Dynamic Cosine Learning Rate Schedule** - Found this in Karpathy Video

(A) **Implimentation of Flash Attention Kernals(v2)** : will speedup the attention part
(A) **Enable cuDNN / cuBLAS auto-tuning** : not very sure what or how it does it, but it speeds up things
(A) **Gradient Accumilation Implimentation** : this will simulate a larger batch size without increasing the vram : simple to impliment also
(E) **Use torch.utils.checkpoint** : Saves ~30–40% memory at the cost of ~15–20% slower compute : used when training large llm

(B) **Pause Training Feature** : Implimnet a way to pause training mid way just to analyse output or change something else, and later on resume when the work is done
(B) **Implimentatoin of Moving Average Train/Val Loss** : In the training loop, other than min, max, and curr training loss, we should also print, a moving avg loss(also I thing we should remove the moving avg loss)
(B) **Upgrade the Save and Load function** : save and load function should be robust, save with apt names, and upgraded to our current model, it should also be able to save the options chozen for the model in a viewable format, and one forlder for each model
(C) **Understand all the Deepseek MLA things** : there are a lot of things implimented in the deepseek artictecture for mla, and not just simple mla, I need to understand each concept in detail.
(C) **Batch size investigation** : Investigate why decreasing the batch size sometime decreases the overall training time.

(D) **Complete control over memory usage** : use of del statments and torch.clear_cache to understand exactly where my memory is going and squeeze the max of my gpu
(B) **Sentence Splitting Implimentaiton** : when loading the training data, the model should be trainied to load from the start of teh sentence. letting it train from the mid of the sentence breaks continuity in it and probably also does not train positional encoding.

(B) **try out bert-based-cased tokenizer** : english only 30k token model
(B) **Try out muon optimizer** : probably faster in convergence than the AdamOptim

(C) **OCR/CNN like kernal shrinker** : for context shrinking and thus fitting in larger contexts, on the tokens because sentence and paragraph tokens also exists na.

(D) **Read Samsung TRM paper** : Tiny Recursive 7Billion parameter model focusing on speciallized use cases

(B) **Impliment Profiling** : Try out torch profiling, will be of great help later on, to understand where is my gpu usage and time are going exactly