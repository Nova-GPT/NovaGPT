- 100M BATCH 8 BLOCK 512 16.77min 56.86days-for-2B-tokens
- 100M BATCH 8 BLOCK 512 11.82min 40.07days-for-2B-tokens 1654MB 7756MB (peak) allocated 8512MB reserved

- 22.2s
- 10.2s

- WGQA-8 16 head 16 layer - perplexity implimentation (95M param)
    1412.382208 MB allocated
    8919.369216 MB peak allocated
    9753.853952 MB reserved
    Total Training Time : 12.06 minutes

- WGQA-8 16 head 18 layer - perplexity implimentation (95M param)
    1412.382208 MB allocated
    8919.369216 MB peak allocated
    9753.853952 MB reserved
    Total Training Time : 12.06 minutes

- WGQA-8 16 head 18 layer - perplexity implimentation (103M param)
    1470.321152 MB allocated
    10227.390976 MB peak allocated
    10525.605888 MB reserved
    Total Training Time : 15.58 minutes

- MLA 16 head 8 layer - 103M param (76M param)
    1696.742912 MB allocated
    6963.769856 MB peak allocated
    7746.879488 MB reserved
    Total Training Time : 4.31 minutes




- MHA 516Embed 12Heads 16Layers :aot_eager 
- 1702.7MB 7804.9MB(peak) allocated 8638.1MB reserved
- Total Training Time : 13.01 minutes

## MHA 516Embed 12Heads 16Layers :default 
- 1695.1MB 5558.3MB(peak) allocated, 5893.0MB reserved
- Total Training Time : 16.79 minutes

## MHA 516Embed 12Heads 16Layers :reduce-overhead:inductor 
- forwardPass 0.0s backwardPassScale 0.0s backwardPassStep 6.2s backwardPassUpdate 0.0s
- 832.5MB 5885.0MB(peak) allocated, 6341.8MB reserved
- Total Training Time : 15.12 minutes

## MHA 516Embed 12Heads 16Layers :gradient-accumilation
- BATCH 16 ITER 100 ACCUMILATION 1 : ~16min
- BATCH 16 ITER 100 ACCUMILATION 2 : OOM Error
- BATCH 08 ITER 200 ACCUMILATION 2 : 6.54min 1423.5MB 5743.7MB(peak) allocated, 6157.2MB reserved
- BATCH 08 ITER 200 ACCUMILATION 4 : 6.89min 1423.5MB 5743.7MB(peak) allocated, 6157.2MB reserved
- BATCH 08 ITER 200 ACCUMILATION 8 : ~7min forwardbackwardPass 16.8s backwardPassStep 0.3s backwardPassUpdate 0.0s

## iter #99
- 802.5MB 5817.1MB(peak) allocated, 6178.2MB reserved
- Total Training Time : 9.90 minutes

## MHA 512Embed 12Heads 16Layers :gradient-accumilation :
- no-sqda 5.27min forwardbackwardPass 2.8s backwardPassStep 0.1s backwardPassUpdate 0.0s 1448.4MB 4803.8MB(peak) allocated, 5207.2MB reserved
- sqda 6.19 minutes 1448.4MB 4845.2MB(peak) allocated, 5677.0MB reserved] 

## 1345.5MB 3120.2MB(peak) allocated, 3741.3MB reserved
- Total Training Time : 6.47 minutes

## Batch size 2 accumilation 2 mla block size 512 embed 516
- 1306.3MB 3028.1MB(peak) allocated, 3231.7MB reserved
- Total Training Time : 1.24 minutes

## MLA batch 4 accumilation 2 mla block size 512 embed 516
- forwardbackwardPass 2.0s backwardPassStep 0.1s backwardPassUpdate 0.0s
- BATCH_SIZE = 4

## MLA batch 2 accumlation mla block size 512 embed 516
- forwardbackwardPass 0.7s backwardPassStep 0.1s backwardPassUpdate 0.0s
- 1301.0MB 3025.9MB(peak) allocated, 3439.3MB reserved
- Total Training Time : 1.12 minutes

## MLA batch 1 accumlation 8 mla block size 512 embed 516
- forwardbackwardPass 1.0s backwardPassStep 0.1s backwardPassUpdate 0.0s
- 1159.5MB 2099.8MB(peak) allocated, 2300.6MB reserved
- Total Training Time : 1.62 minutes

## MLA batch 1 accumlation 4 mla block size 512 embed 516
- forwardbackwardPass 0.5s backwardPassStep 0.1s backwardPassUpdate 0.0s
- 1159.2MB 2094.4MB(peak) allocated, 2300.6MB reserved
- Total Training Time : 1.69 minutes


