from tqdm import tqdm
from collections import Counter, deque
import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPEtokenizer:
  def __init__(self):
    self.vocab = {}
    self.inverse_vocab = {}
    self.bpe_merges = {}
 
  def train_BPE(self, text, vocab_size, allowed_specials = {"<|endoftext|>"}):
    unique_chars = [ chr(i) for i in range(256)]
    
    unique_chars.extend(char for char in sorted(set(text)) if char not in unique_chars)

    self.vocab = {i:char for i,char in enumerate(unique_chars)}
    self.inverse_vocab = {char : i for i, char in self.vocab.items()}
    
    if allowed_specials:
      for token in allowed_specials:
        if token not in self.inverse_vocab:
          new_id = len(self.vocab)
          self.inverse_vocab[token] = new_id
          self.vocab[new_id] = token
    
    token_ids = []
    for char in text:
      token = self.inverse_vocab[char]
      token_ids.append(token)
        
    #================BPE Algorithm================
    for new_id in tqdm(range(len(self.vocab), vocab_size), desc="Training BPE"):
        pair_id = self.find_freq_pair(token_ids)
        
        if pair_id is None: #No more pair to merge; Stop the training
            break
        token_ids = self.replace_pairs(token_ids, pair_id, new_id)
        self.bpe_merges[pair_id] = new_id
    
    #================Update the vocabulary with the new merged_tokens================
    for (p0, p1), new_id in self.bpe_merges.items():
        merged_token = self.vocab[p0] + self.vocab[p1]
        self.vocab[new_id] = merged_token
        self.inverse_vocab[merged_token] = new_id
    
    return
  
  @staticmethod
  def find_freq_pair(token_ids):
      pairs = Counter(zip(token_ids, token_ids[1:]))
      
      if not pairs:
          return None

      return max(pairs.items(), key = lambda x : x[1])[0]
  
  @staticmethod
  def replace_pairs(token_ids, pair_id, new_id):
      dq = deque(token_ids)
      replaced = []

      while dq:
          current = dq.popleft()
          if dq and (current, dq[0]) == pair_id:
              replaced.append(new_id)
              # Remove the 2nd token of the pair, 1st was already removed
              dq.popleft()
          else:
              replaced.append(current)

      return replaced
  
  def encode(self, text:str):
        """ 
            This function will generate token_ids based on the BPE merge rules and the vocabularies it learned during the training   
        """
        tokens = []
        
        #Split the text into tokens, Keeping the newline intact
        words = text.replace("\n", " \n ").split()  #Make sure that the new_line seperator "\n" is treated as a separate token
        
        for i, word in enumerate(words):
          tokens.append(word)
                
        token_ids = []
        for token in tokens:
            #Check if the token is already present in the vocabulary or not
            if token in self.inverse_vocab:
                token_id = self.inverse_vocab[token]
                token_ids.append(token_id)
            else:
                #Do subword_tokenization using BPE
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)
                
        return token_ids

  #=======================This function is used to tokenize a sinlge token using BPE merge rules=======================                   
  def tokenize_with_bpe(self, token):
      
      #Tokenize the tokens into individual characters(it can be interpreted as initial token_Ids)
      token_ids = [self.inverse_vocab.get(char, None) for char in token]
      if None in token_ids:
          missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
          raise ValueError(f"Characters not found in vocabulary: {missing_chars}")
      
      can_merge = True
      while can_merge and len(token_ids) > 1:
          can_merge = False
          new_tokens = []
          i = 0
          while i < len(token_ids) - 1:
              pair = (token_ids[i] , token_ids[i+1])
              if pair in self.bpe_merges:
                  merged_token_id = self.bpe_merges[pair]
                  new_tokens.append(merged_token_id)
                  
                  i += 2 #Skip the next token as it is already merged
                  can_merge = True
              else:
                  new_tokens.append(token_ids[i])
                  i+=1
          if i < len(token_ids):
              new_tokens.append(token_ids[i])
          token_ids = new_tokens
              
      return token_ids
                  
      
  #======================This function is used to decode a list of token_ids back to text======================
  def decode(self, token_ids):
      decoded_string = ""
      for token_id in token_ids:
          if token_id not in self.vocab:
              raise ValueError(f"Token Id {token_id} not found in Vocabulary")
          token = self.vocab[token_id]
          if token.startswith("_"):
              #Replace the "_" with space
              decoded_string += " " + token[1:]
          else:
              decoded_string += token
          
      return decoded_string 
                
        
def main():
  tokenizer = BPEtokenizer()
    
  #test_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
  file_path = r"C:\Users\Ihor\Desktop\Stanford_LLM\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
  with open(file_path, encoding="utf-8") as file:
    data = file.read()
    print(f"Length of the text: {len(data)}")
  re.findall(PAT, data)
    
  tokenizer.train_BPE(text=data, vocab_size=1000, allowed_specials={"<|endoftext|>"})
  
  random_text = "Jack embraced beauty through art and life"
  token_ids = tokenizer.encode(random_text)
  print(token_ids)
  print(tokenizer.decode(token_ids))
  
  print(f"Number of characters in random_text: {len(random_text)}")
  print(f"Number of token Ids:- {len(token_ids)}")
  
  for token_id in token_ids:
    print(f"{token_id}-->{tokenizer.decode([token_id])}")
  
if __name__ == "__main__":
  main()