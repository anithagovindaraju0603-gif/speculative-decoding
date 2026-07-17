import torch 

def rejection_sample(
    draft_tokens: torch.Tensor,   # [K] int64
    draft_probs: torch.Tensor,    # [K, V] float32, already at temp T
    target_probs: torch.Tensor,   # [K+1, V] float32, already at temp T
) -> tuple[int, torch.Tensor]:
    
    K = draft_tokens.shape[0] 

    for i in range(K):
        token = draft_tokens[i].item()          # scalar int
        p_i = target_probs[i, token].item()     # scalar, target_probs contain probs of all vocab, token is needed to filter out the prob of the specific token
        q_i = draft_probs[i, token].item()      # scalar

        r = torch.rand(1).item()
        if r < (p_i/q_i): #same as min(1.0, p_i / q_i)
            continue  # accept, move to next draft
        else:
            # Reject: sample replacement from normalize(max(0, p - q)) 
            #example: Vocab of 4 tokens: [A, B, C, D] --> q: [0.7, 0.1, 0.1, 0.1], p: [0.2, 0.3, 0.3, 0.2]. Here for A, q(0.7) is way more confident than 0.2
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0.0) # [0.2 - 0.7, 0.3 - 0.1, 0.3 - 0.1, 0.2 - 0.1] = [-0.5, 0.2, 0.2, 0.1] ---> clamp(min=0) — kill negatives = [0, 0.2, 0.2, 0.1]
            adjusted = adjusted / adjusted.sum() # normalize and divide by 0.5 = [0, 0.4, 0.4, 0.2] #valid distribution summing to 1.
            replacement = torch.multinomial(adjusted, num_samples=1)  # [1]
            return i, replacement

    # All K accepted — sample bonus from target's K+1-th distribution
    bonus = torch.multinomial(target_probs[-1], num_samples=1)  # [1]
    return K, bonus