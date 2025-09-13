"""
Tokenizer for Supernova USLM - Handles text preprocessing and tokenization
"""

import torch
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer
import sentencepiece as spm
import json
import os


class SupernovaTokenizer:
    """Tokenizer wrapper for Supernova USLM model"""
    
    def __init__(self, tokenizer_path: Optional[str] = None, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        
        # Special tokens
        self.special_tokens = {
            'pad_token': '<pad>',
            'unk_token': '<unk>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'user_token': '<|user|>',
            'assistant_token': '<|assistant|>',
            'system_token': '<|system|>',
        }
        
        # Token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.user_token_id = 4
        self.assistant_token_id = 5
        self.system_token_id = 6
        
        # Load or create tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load_tokenizer(tokenizer_path)
        else:
            # Use a pretrained tokenizer as base (LLaMA-style)
            try:
                self.base_tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/DialoGPT-medium", 
                    trust_remote_code=True
                )
                print("Loaded base tokenizer: microsoft/DialoGPT-medium")
            except:
                try:
                    # Fallback to a simpler tokenizer
                    self.base_tokenizer = AutoTokenizer.from_pretrained(
                        "gpt2", 
                        trust_remote_code=True
                    )
                    print("Loaded fallback tokenizer: gpt2")
                except:
                    raise RuntimeError("Could not load any base tokenizer. Please install transformers properly.")
            
            # Add special tokens
            self.base_tokenizer.add_special_tokens({
                'pad_token': self.special_tokens['pad_token'],
                'unk_token': self.special_tokens['unk_token'],
                'bos_token': self.special_tokens['bos_token'],
                'eos_token': self.special_tokens['eos_token'],
                'additional_special_tokens': [
                    self.special_tokens['user_token'],
                    self.special_tokens['assistant_token'],
                    self.special_tokens['system_token'],
                ]
            })
            
            # Update vocab size
            self.actual_vocab_size = len(self.base_tokenizer)
            print(f"Tokenizer vocab size: {self.actual_vocab_size}")
    
    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token ids"""
        if hasattr(self, 'base_tokenizer'):
            tokens = self.base_tokenizer.encode(
                text, 
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=max_length is not None,
                return_tensors=None
            )
            return tokens
        else:
            # Custom tokenizer implementation would go here
            return [self.unk_token_id] * len(text.split())
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if hasattr(self, 'base_tokenizer'):
            return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            # Custom decode implementation would go here
            return f"[Decoded {len(token_ids)} tokens]"
    
    def batch_encode(self, texts: List[str], max_length: Optional[int] = None, padding: bool = True) -> Dict[str, torch.Tensor]:
        """Batch encode multiple texts"""
        if hasattr(self, 'base_tokenizer'):
            return self.base_tokenizer(
                texts,
                max_length=max_length,
                padding=padding,
                truncation=max_length is not None,
                return_tensors="pt"
            )
        else:
            # Custom implementation
            encoded = [self.encode(text, max_length=max_length) for text in texts]
            max_len = max(len(seq) for seq in encoded)
            
            # Pad sequences
            padded = []
            attention_masks = []
            for seq in encoded:
                padding_length = max_len - len(seq)
                padded_seq = seq + [self.pad_token_id] * padding_length
                attention_mask = [1] * len(seq) + [0] * padding_length
                padded.append(padded_seq)
                attention_masks.append(attention_mask)
            
            return {
                'input_ids': torch.tensor(padded, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
    
    def format_chat_prompt(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages for chat completion"""
        formatted_prompt = ""
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_prompt += f"{self.special_tokens['system_token']} {content}\n"
            elif role == 'user':
                formatted_prompt += f"{self.special_tokens['user_token']} {content}\n"
            elif role == 'assistant':
                formatted_prompt += f"{self.special_tokens['assistant_token']} {content}\n"
        
        if add_generation_prompt:
            formatted_prompt += f"{self.special_tokens['assistant_token']} "
        
        return formatted_prompt
    
    def save_tokenizer(self, save_path: str):
        """Save tokenizer configuration"""
        config = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'tokenizer_type': 'supernova'
        }
        
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'tokenizer_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        if hasattr(self, 'base_tokenizer'):
            self.base_tokenizer.save_pretrained(save_path)
        
        print(f"Tokenizer saved to {save_path}")
    
    def load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer from saved configuration"""
        config_path = os.path.join(tokenizer_path, 'tokenizer_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.vocab_size = config.get('vocab_size', self.vocab_size)
            self.special_tokens.update(config.get('special_tokens', {}))
        
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}")
        except:
            print(f"Could not load tokenizer from {tokenizer_path}")
    
    def __len__(self):
        """Return vocabulary size"""
        if hasattr(self, 'base_tokenizer'):
            return len(self.base_tokenizer)
        return self.vocab_size
    
    @property
    def vocab_size_actual(self):
        """Get actual vocabulary size"""
        return len(self)


def create_chat_dataset_entry(user_message: str, assistant_message: str, system_message: Optional[str] = None) -> Dict[str, str]:
    """Helper function to create a chat dataset entry"""
    messages = []
    
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_message})
    
    return {"messages": messages}


def format_training_example(tokenizer: SupernovaTokenizer, messages: List[Dict[str, str]], max_length: int = 2048) -> Dict[str, torch.Tensor]:
    """Format a training example for conversation fine-tuning"""
    # Format the conversation
    formatted_text = tokenizer.format_chat_prompt(messages, add_generation_prompt=False)
    
    # Encode
    encoded = tokenizer.encode(formatted_text, max_length=max_length)
    
    # Create labels (same as input_ids for causal LM)
    labels = encoded.copy()
    
    # Mask user and system tokens in labels (only train on assistant responses)
    input_ids_tensor = torch.tensor(encoded)
    labels_tensor = torch.tensor(labels)
    
    # Find assistant response start positions
    assistant_token_id = tokenizer.assistant_token_id if hasattr(tokenizer, 'assistant_token_id') else tokenizer.encode(tokenizer.special_tokens['assistant_token'])[0]
    
    # Mask everything except assistant responses
    mask_labels = True
    for i, token_id in enumerate(input_ids_tensor):
        if token_id == assistant_token_id:
            mask_labels = False
        elif token_id in [tokenizer.user_token_id, tokenizer.system_token_id] if hasattr(tokenizer, 'user_token_id') else []:
            mask_labels = True
        
        if mask_labels:
            labels_tensor[i] = -100  # Ignore in loss computation
    
    # Pad if necessary
    if len(encoded) < max_length:
        padding_length = max_length - len(encoded)
        input_ids_tensor = torch.cat([input_ids_tensor, torch.full((padding_length,), tokenizer.pad_token_id)])
        labels_tensor = torch.cat([labels_tensor, torch.full((padding_length,), -100)])
        attention_mask = torch.cat([torch.ones(len(encoded)), torch.zeros(padding_length)])
    else:
        attention_mask = torch.ones(len(encoded))
    
    return {
        'input_ids': input_ids_tensor,
        'labels': labels_tensor,
        'attention_mask': attention_mask
    }


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = SupernovaTokenizer()
    
    # Test basic encoding/decoding
    text = "Hello, how are you today?"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test chat formatting
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."}
    ]
    
    formatted = tokenizer.format_chat_prompt(messages)
    print(f"\nChat format:\n{formatted}")
    
    # Test training example formatting
    training_example = format_training_example(tokenizer, messages, max_length=128)
    print(f"\nTraining example shapes:")
    for key, value in training_example.items():
        print(f"{key}: {value.shape}")
