import torch

def custom_tokenizer_torch(text):
    # Convert entire text to ASCII codes tensor on GPU
    words_tensor = torch.tensor([ord(c) for c in text if 32 <= ord(c) <= 126], dtype=torch.int32).cuda()

    # Encoding / decoding dictionaries start with ASCII chars
    encode_dict = {chr(i): i for i in range(0, 127)}
    decode_dict = {v: k for k, v in encode_dict.items()}
    next_num = 127

    def find_most_common_pair(words_tensor):
        if words_tensor.size(0) < 2:
            return None
        pairs = torch.stack([words_tensor[:-1], words_tensor[1:]], dim=1)
        unique_pairs, inverse_indices = torch.unique(pairs, dim=0, return_inverse=True)
        counts = torch.bincount(inverse_indices)
        max_count, max_idx = torch.max(counts, dim=0)
        if max_count < 2:
            return None
        return unique_pairs[max_idx][0].item(), unique_pairs[max_idx][1].item()

    def replace_pair(words_list, pair, new_token):
        new_list = []
        i = 0
        while i < len(words_list):
            if i < len(words_list) - 1 and (words_list[i], words_list[i+1]) == pair:
                new_list.append(new_token)
                i += 2
            else:
                new_list.append(words_list[i])
                i += 1
        return new_list

    # Initially, work with a Python list for replacement, GPU accelerated only for counting pairs
    words_list = words_tensor.cpu().tolist()
    i = 0
    while True:
        i+=1
        print(i, end = " ")
        # Use GPU tensor for counting
        temp_tensor = torch.tensor(words_list, dtype=torch.int32).cuda()
        pair = find_most_common_pair(temp_tensor)
        if pair is None:
            break

        # Decode pair to string for new token key
        conact = decode_dict.get(pair[0], chr(pair[0])) + decode_dict.get(pair[1], chr(pair[1]))
        encode_dict[conact] = next_num
        decode_dict[next_num] = conact

        # Replace pair in Python list
        words_list = replace_pair(words_list, pair, next_num)
        next_num += 1

    return words_list, encode_dict, decode_dict

# Example usage
print("here")
text = open('input.txt', 'r', encoding='utf-8').read()
tokens, enc_dict, dec_dict = custom_tokenizer_torch(text)

print("Tokens (sample):", tokens[:100])
print("Encoding dictionary sample:", {k: enc_dict[k] for k in list(enc_dict)})
print("Decoding dictionary sample:", {k: dec_dict[k] for k in list(dec_dict)})
