# --- Copilot test commit: This comment is for git commit demonstration. ---
"""
Training Pipeline for Supernova USLM - Conversation Fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import torch.nn.functional as F

import json
import os
import logging
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import random
import numpy as np
import math

from supernova_model import SupernovaForCausalLM, SupernovaConfig, create_supernova_model
from tokenizer import SupernovaTokenizer, format_training_example
from chat_interface import SupernovaChat


@dataclass
class TrainingConfig:
    """Training configuration for Supernova USLM"""
    
    # Model parameters
    model_name: str = "supernova-chat"
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_epochs: int = 15
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    
    # Data parameters
    max_sequence_length: int = 1024
    train_data_path: str = "train_data.json"
    eval_data_path: Optional[str] = None
    data_split_ratio: float = 0.9
    
    # Optimization
    gradient_clipping: float = 1.0
    mixed_precision: bool = True
    compile_model: bool = False
    
    # Evaluation and saving
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    output_dir: str = "outputs"
    save_total_limit: int = 3
    
    # Conversation-specific
    mask_user_tokens: bool = True
    system_message_weight: float = 0.1
    response_loss_weight: float = 1.0
    
    # Hardware
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42


class ConversationDataset(Dataset):
    """Dataset for conversation fine-tuning"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: SupernovaTokenizer, 
        max_length: int = 1024,
        mask_user_tokens: bool = True,
        system_message_weight: float = 0.1
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_user_tokens = mask_user_tokens
        self.system_message_weight = system_message_weight
        
        # Load conversation data
        self.conversations = self.load_conversations(data_path)
        
        logging.info(f"Loaded {len(self.conversations)} conversations from {data_path}")
    
    def load_conversations(self, data_path: str) -> List[Dict]:
        """Load conversation data from file"""
        conversations = []
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    conversations = data
                elif isinstance(data, dict) and 'conversations' in data:
                    conversations = data['conversations']
                else:
                    # Single conversation format
                    conversations = [data]
        
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    conversations.append(json.loads(line.strip()))
        
        # Filter and validate conversations
        valid_conversations = []
        for conv in conversations:
            if self.validate_conversation(conv):
                valid_conversations.append(conv)
        
        return valid_conversations
    
    def validate_conversation(self, conversation: Dict) -> bool:
        """Validate conversation format"""
        if 'messages' not in conversation:
            return False
        
        messages = conversation['messages']
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        
        # Check for at least one user and one assistant message
        has_user = any(msg.get('role') == 'user' for msg in messages)
        has_assistant = any(msg.get('role') == 'assistant' for msg in messages)
        
        return has_user and has_assistant
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.conversations[idx]
        messages = conversation['messages']
        
        # Format the training example
        formatted_example = format_training_example(
            self.tokenizer, 
            messages, 
            max_length=self.max_length
        )
        
        # Apply conversation-specific modifications
        if self.mask_user_tokens:
            formatted_example = self.apply_conversation_masking(
                formatted_example, messages
            )
        
        return formatted_example
    
    def apply_conversation_masking(
        self, 
        example: Dict[str, torch.Tensor], 
        messages: List[Dict[str, str]]
    ) -> Dict[str, torch.Tensor]:
        """Apply conversation-specific loss masking and guarantee at least one unmasked label."""
        input_ids = example['input_ids']
        labels = example['labels'].clone()

        # Create a more sophisticated masking strategy
        # We want to train primarily on assistant responses
        full_text = self.tokenizer.format_chat_prompt(messages, add_generation_prompt=False)
        current_pos = 0
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                role_text = f"{self.tokenizer.special_tokens['system_token']} {content}\n"
                role_length = len(self.tokenizer.encode(role_text, add_special_tokens=False))
                for i in range(current_pos, min(current_pos + role_length, len(labels))):
                    if labels[i] != -100:
                        if random.random() > self.system_message_weight:
                            labels[i] = -100
                current_pos += role_length
            elif role == 'user':
                role_text = f"{self.tokenizer.special_tokens['user_token']} {content}\n"
                role_length = len(self.tokenizer.encode(role_text, add_special_tokens=False))
                for i in range(current_pos, min(current_pos + role_length, len(labels))):
                    labels[i] = -100
                current_pos += role_length
            elif role == 'assistant':
                role_text = f"{self.tokenizer.special_tokens['assistant_token']} {content}\n"
                role_length = len(self.tokenizer.encode(role_text, add_special_tokens=False))
                current_pos += role_length

        # Guarantee at least one unmasked label (not -100)
        if (labels != -100).sum().item() == 0:
            # Unmask the last non-padding token
            nonpad_indices = (input_ids != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(nonpad_indices) > 0:
                last_idx = nonpad_indices[-1].item()
                labels[last_idx] = input_ids[last_idx]
            else:
                labels[-1] = input_ids[-1]

        example['labels'] = labels
        return example


def create_sample_training_data(output_path: str = "sample_train_data.json", num_samples: int = 100):
    """Create sample training data for conversation fine-tuning"""
    
    sample_conversations = []
    
    # Different types of conversations to train on
    conversation_types = [
        # Q&A conversations
        {
            "system": "You are a helpful, harmless, and honest AI assistant.",
            "examples": [
                ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."),
                ("How does photosynthesis work?", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll in their leaves."),
                ("What are the benefits of exercise?", "Regular exercise improves cardiovascular health, strengthens muscles and bones, boosts mental health, helps maintain healthy weight, and increases energy levels."),
                ("Explain quantum computing simply.", "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that could potentially solve certain problems much faster than classical computers."),
                ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities that increase greenhouse gas concentrations in the atmosphere.")
            ]
        },
        
        # Helpful assistant conversations
        {
            "system": "You are Supernova, a helpful AI assistant created to assist users with various tasks.",
            "examples": [
                ("Can you help me write a professional email?", "I'd be happy to help you write a professional email. Could you tell me the purpose of the email and any key points you'd like to include?"),
                ("I need to organize my day better.", "I can help you with time management! Consider using techniques like time-blocking, the Pomodoro technique, or creating a prioritized to-do list. What specific challenges are you facing with organization?"),
                ("How do I learn a new skill effectively?", "To learn a new skill effectively: 1) Set clear, specific goals, 2) Break it into smaller steps, 3) Practice consistently, 4) Seek feedback, 5) Apply what you learn in real situations, and 6) Be patient with yourself."),
                ("What's a good way to start coding?", "Start with a beginner-friendly language like Python. Focus on fundamentals: variables, loops, functions, and problem-solving. Practice with small projects, use online tutorials, and code regularly. What interests you most about programming?"),
                ("I'm feeling overwhelmed with work.", "Feeling overwhelmed is common. Try breaking large tasks into smaller ones, prioritizing urgent items, taking regular breaks, and don't hesitate to ask for help when needed. What specific aspects of work are causing the most stress?")
            ]
        },
        
        # Creative conversations
        {
            "system": "You are a creative and imaginative AI assistant who enjoys helping with creative tasks.",
            "examples": [
                ("Write a short poem about the ocean.", "Waves whisper secrets to the shore,\nEndless blue beneath sky's dome,\nSalt and spray, ancient lore,\nThe ocean calls—we're coming home."),
                ("Give me a creative story idea.", "A librarian discovers that certain books in their collection are portals to the worlds described within them, but each visit changes the story slightly, affecting both the book world and reality."),
                ("Help me brainstorm names for my coffee shop.", "Here are some creative coffee shop names: 'The Daily Grind,' 'Bean There, Done That,' 'Espresso Yourself,' 'The Roasted Bean,' 'Caffeine & Dreams,' 'The Steaming Cup,' or 'Brew & You.'"),
                ("What's a fun weekend project?", "Try creating a mini herb garden on your windowsill, learning to make homemade pasta from scratch, starting a photo journal of your neighborhood, or building a reading nook in your home."),
                ("I want to start a creative hobby.", "Consider these creative hobbies: watercolor painting, creative writing, pottery, photography, knitting, woodworking, digital art, or learning a musical instrument. What type of creativity appeals to you most?")
            ]
        }
    ]
    
    for conv_type in conversation_types:
        system_msg = conv_type["system"]
        for user_msg, assistant_msg in conv_type["examples"]:
            conversation = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            }
            sample_conversations.append(conversation)
    
    # Add some multi-turn conversations
    multi_turn_conversations = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "I'm learning Python. Can you explain what a list is?"},
                {"role": "assistant", "content": "A list in Python is an ordered collection of items that can hold different types of data. You create a list using square brackets, like [1, 2, 3] or ['apple', 'banana', 'cherry']. Lists are mutable, meaning you can change their contents after creation."},
                {"role": "user", "content": "How do I add items to a list?"},
                {"role": "assistant", "content": "You can add items to a list in several ways:\n1. Use `.append()` to add one item to the end: my_list.append('new_item')\n2. Use `.insert()` to add at a specific position: my_list.insert(0, 'first_item')\n3. Use `.extend()` to add multiple items: my_list.extend(['item1', 'item2'])\n4. Use `+` operator to combine lists: new_list = my_list + ['item1', 'item2']"},
                {"role": "user", "content": "Thanks! That's very helpful."},
                {"role": "assistant", "content": "You're welcome! Feel free to ask if you have more questions about Python lists or any other programming concepts. Practice with these methods and you'll get comfortable with list manipulation quickly!"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Supernova, a knowledgeable AI assistant."},
                {"role": "user", "content": "What's the difference between weather and climate?"},
                {"role": "assistant", "content": "Weather refers to short-term atmospheric conditions in a specific place at a specific time - like today's temperature, rain, or wind. Climate refers to long-term patterns of weather averaged over many years (typically 30+ years) in a region."},
                {"role": "user", "content": "So if it's unusually cold today, that doesn't mean climate change isn't happening?"},
                {"role": "assistant", "content": "Exactly right! One cold day, or even a cold week, doesn't contradict climate change. Climate scientists look at long-term trends across decades and globally. Short-term weather variations are normal and expected, while climate change refers to shifts in these long-term average patterns."},
                {"role": "user", "content": "That makes sense. What evidence do we have for climate change?"},
                {"role": "assistant", "content": "Key evidence includes: 1) Global temperature records showing warming trends, 2) Melting ice sheets and glaciers, 3) Rising sea levels, 4) Changes in precipitation patterns, 5) Earlier spring events like flowering and bird migration, 6) Ocean acidification, and 7) Ice core data showing current CO2 levels are the highest in 800,000 years."}
            ]
        }
    ]
    
    sample_conversations.extend(multi_turn_conversations)
    
    # Shuffle and expand the dataset
    all_conversations = sample_conversations * (num_samples // len(sample_conversations) + 1)
    random.shuffle(all_conversations)
    sample_conversations = all_conversations[:num_samples]
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample training data with {len(sample_conversations)} conversations saved to {output_path}")
    return output_path


class SupernovaTrainer:
    """Trainer for Supernova USLM conversation fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create output directory first
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds
        self.set_seed(config.seed)
        
        # Setup device
        self.device = self._setup_device(config.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.eval_loader = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        
        # Load tokenizer
        if self.config.tokenizer_path:
            self.tokenizer = SupernovaTokenizer(self.config.tokenizer_path)
        else:
            self.tokenizer = SupernovaTokenizer()
        
        self.logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        
        # Load or create model
        if self.config.model_path and os.path.exists(self.config.model_path):
            self.logger.info(f"Loading model from {self.config.model_path}")
            # Load saved model
            config = SupernovaConfig()
            self.model = SupernovaForCausalLM(config)
            checkpoint = torch.load(
                os.path.join(self.config.model_path, "pytorch_model.bin"), 
                map_location="cpu"
            )
            self.model.load_state_dict(checkpoint)
        else:
            self.logger.info("Creating new Supernova model")
            # Create model with full vocab size
            config = SupernovaConfig()
            config.vocab_size = len(self.tokenizer)
            config.hidden_size = 768
            config.num_hidden_layers = 12
            config.num_attention_heads = 12
            config.num_key_value_heads = 12
            config.intermediate_size = 3072
            config.hidden_act = "swiglu"
            config.max_position_embeddings = 2048
            config.initializer_range = 0.02
            config.rms_norm_eps = 1e-6
            config.use_cache = True
            config.tie_word_embeddings = True
            config.attention_bias = False
            config.attention_dropout = 0.1
            config.hidden_dropout = 0.1
            config.device = self.device
            config.dtype = torch.float32
            self.model = create_supernova_model(vocab_size=config.vocab_size,
                                              hidden_size=config.hidden_size,
                                              num_layers=config.num_hidden_layers,
                                              num_heads=config.num_attention_heads,
                                              num_kv_heads=config.num_key_value_heads,
                                              intermediate_size=config.intermediate_size,
                                              max_position_embeddings=config.max_position_embeddings)
        
        # Move model to device
        self.model.to(self.device)
        
        # Compile model if requested
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.logger.info("Compiling model...")
            self.model = torch.compile(self.model)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model loaded: {total_params:,} total parameters")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_datasets(self):
        """Setup training and evaluation datasets"""
        
        # Create sample data if it doesn't exist
        if not os.path.exists(self.config.train_data_path):
            self.logger.info(f"Creating sample training data at {self.config.train_data_path}")
            create_sample_training_data(self.config.train_data_path, num_samples=500)
        
        # Load training dataset
        train_dataset = ConversationDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_sequence_length,
            mask_user_tokens=self.config.mask_user_tokens,
            system_message_weight=self.config.system_message_weight
        )
        
        # Split into train/eval if no separate eval data
        if self.config.eval_data_path and os.path.exists(self.config.eval_data_path):
            eval_dataset = ConversationDataset(
                data_path=self.config.eval_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_sequence_length,
                mask_user_tokens=self.config.mask_user_tokens,
                system_message_weight=self.config.system_message_weight
            )
        else:
            # Split the training dataset
            train_size = int(len(train_dataset) * self.config.data_split_ratio)
            eval_size = len(train_dataset) - train_size
            
            train_dataset, eval_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, eval_size]
            )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn
        )
        
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for data loader.
        Stacks tensors and performs a batch-level validation check:
        Warns if all labels in a batch are masked, which would indicate a data or masking bug.
        """
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        # Data validation: warn if all labels are masked in any batch
        # This check helps catch silent failures in the data pipeline or masking logic.
        if (labels != -100).sum().item() == 0:
            print('Warning: All labels are masked in a batch!')
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        
        # Initialize weights with a smaller range
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Use Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_normal_(module.weight.data, gain=0.1)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        self.model.apply(init_weights)
        
        # Scale down initial embeddings for better loss scaling
        if hasattr(self.model, 'embed_tokens'):
            with torch.no_grad():
                self.model.embed_tokens.weight.data.mul_(0.1)
        
        # Setup optimizer with parameter-specific settings
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.95),  # Modified betas for stability
            weight_decay=self.config.weight_decay
        )
        
        # Calculate total training steps once
        total_steps = len(self.train_loader) * self.config.max_epochs
        if self.config.max_steps:
            total_steps = min(total_steps, self.config.max_steps)
        warmup_steps = self.config.warmup_steps
        
        # Create scheduler with linear warmup and cosine decay
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            cycle_momentum=False
        )
        
        # Setup mixed precision
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info(f"Optimizer and scheduler setup complete. Total steps: {total_steps}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        
        self.model.train()
        

        # Move batch to device and ensure float32
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Check number of unmasked labels
        if 'labels' in batch:
            num_unmasked = (batch['labels'] != -100).sum().item()
            if num_unmasked == 0:
                self.logger.warning('All labels are masked in this batch. Skipping batch.')
                return 0.0
            else:
                self.logger.debug(f'Unmasked labels in batch: {num_unmasked}')

        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        try:
            # Forward pass with gradient computation
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            if loss is None:
                return 0.0
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"Found NaN/Inf loss before scaling: {loss.item()}")
                return 0.0
            
            # Scale loss and check again
            loss = loss / self.config.gradient_accumulation_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"Found NaN/Inf loss after scaling: {loss.item()}")
                return 0.0
            

            # Backward pass with gradient scaling
            loss.backward()

            # Check gradients for NaN/Inf and count nonzero gradients
            valid_gradients = True
            nonzero_grad_count = 0
            total_param_count = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    total_param_count += 1
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self.logger.warning(f"Found NaN/Inf gradients in {name}")
                        valid_gradients = False
                        break
                    if param.grad.abs().sum().item() > 0:
                        nonzero_grad_count += 1

            if not valid_gradients:
                self.optimizer.zero_grad(set_to_none=True)
                return 0.0

            if nonzero_grad_count == 0:
                self.logger.warning('All gradients are zero after backward! Skipping optimizer step.')
                self.optimizer.zero_grad(set_to_none=True)
                return 0.0
            else:
                self.logger.debug(f'Nonzero gradients: {nonzero_grad_count}/{total_param_count}')
            
            # Clip gradients
            if self.config.gradient_clipping > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clipping
                )
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.warning(f"Found NaN/Inf gradient norm: {grad_norm}")
                    self.optimizer.zero_grad(set_to_none=True)
                    return 0.0
            
            # Update weights if we've accumulated enough gradients
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                # Only step scheduler if not past total_steps
                if hasattr(self.scheduler, 'total_steps'):
                    if self.scheduler.last_epoch < self.scheduler.total_steps - 1:
                        self.scheduler.step()
                else:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Log gradient norms periodically
            if (self.global_step + 1) % self.config.logging_steps == 0:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.logger.info(f"Gradient norm: {total_norm:.4f}")
            
            return loss.item() * self.config.gradient_accumulation_steps
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            raise e
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch + 1}/{self.config.max_epochs}"
        )
        
        for step, batch in enumerate(progress_bar):
            
            # Training step
            loss = self.train_step(batch)
            total_loss += loss
            num_steps += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                
                # Clip gradients
                if self.config.gradient_clipping > 0:
                    if self.config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clipping
                    )
                
                # Optimizer step
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Only step scheduler if not past total_steps
                if hasattr(self.scheduler, 'total_steps'):
                    if self.scheduler.last_epoch < self.scheduler.total_steps - 1:
                        self.scheduler.step()
                else:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_steps
                    lr = self.scheduler.get_last_lr()[0]
                    
                    self.logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                    )
                    
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    self.logger.info(f"Evaluation loss: {eval_loss:.4f}")
                    
                    # Save best model
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_model("best_model")
                        self.logger.info("New best model saved!")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_model(f"checkpoint-{self.global_step}")
                    # Save training state for resumption
                    self.save_training_state()
                    # Clean up old checkpoints
                    self.cleanup_old_checkpoints()
                
                # Check if max steps reached
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break
        
        return total_loss / num_steps if num_steps > 0 else 0.0
    
    def evaluate(self) -> float:
        """Evaluate the model"""
        
        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.get('loss', None)
                    
                    if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        num_steps += 1
                except Exception as e:
                    self.logger.warning(f"Error during evaluation: {str(e)}")
                    continue
        
        avg_loss = total_loss / max(num_steps, 1)
        self.logger.info(f"Evaluation completed with {num_steps} valid steps")
        return avg_loss
    
    def save_model(self, save_name: str):
        """Save model and tokenizer with comprehensive checkpointing"""
        
        save_path = os.path.join(self.config.output_dir, save_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # Save complete checkpoint with training state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.__dict__,
            'model_config': self.model.config.__dict__ if hasattr(self.model, 'config') else {}
        }
        torch.save(checkpoint, os.path.join(save_path, "training_checkpoint.pt"))
        
        # Save model config
        config_dict = {
            'model_name': self.config.model_name,
            'vocab_size': len(self.tokenizer),
            'training_config': self.config.__dict__,
            'architecture': 'SupernovaForCausalLM',
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'created_at': time.time()
        }
        
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save tokenizer
        tokenizer_path = os.path.join(save_path, "tokenizer")
        self.tokenizer.save_tokenizer(tokenizer_path)
        
        # Save training metrics if available
        if hasattr(self, 'training_metrics'):
            with open(os.path.join(save_path, "metrics.json"), 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
        
        self.logger.info(f"Model and checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint to resume training"""
        
        checkpoint_file = os.path.join(checkpoint_path, "training_checkpoint.pt")
        if not os.path.exists(checkpoint_file):
            self.logger.warning(f"Checkpoint not found at {checkpoint_file}")
            return False
        
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from step {self.global_step}, epoch {self.current_epoch}")
        
        return True
    
    def save_training_state(self):
        """Save current training state for resumption"""
        
        state_path = os.path.join(self.config.output_dir, "latest_state")
        os.makedirs(state_path, exist_ok=True)
        
        # Save lightweight state for quick resumption
        state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_loss': self.best_eval_loss,
            'timestamp': time.time()
        }
        
        with open(os.path.join(state_path, "training_state.json"), 'w') as f:
            json.dump(state, f, indent=2)
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        
        if self.config.save_total_limit <= 0:
            return
        
        # Get all checkpoint directories
        checkpoint_dirs = []
        for item in os.listdir(self.config.output_dir):
            item_path = os.path.join(self.config.output_dir, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                try:
                    step = int(item.split('-')[1])
                    checkpoint_dirs.append((step, item_path))
                except:
                    continue
        
        # Sort by step number and keep only the latest ones
        checkpoint_dirs.sort(key=lambda x: x[0])
        
        # Remove old checkpoints
        while len(checkpoint_dirs) > self.config.save_total_limit:
            _, old_path = checkpoint_dirs.pop(0)
            try:
                import shutil
                shutil.rmtree(old_path)
                self.logger.info(f"Removed old checkpoint: {old_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {old_path}: {e}")
    
    def train(self):
        """Main training loop"""
        
        self.logger.info("Starting training...")
        
        # Setup everything
        self.setup_model_and_tokenizer()
        self.setup_datasets()
        self.setup_optimizer_and_scheduler()
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            epoch_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss:.4f}")
            
            # Final evaluation
            eval_loss = self.evaluate()
            self.logger.info(f"Epoch {epoch + 1} evaluation loss: {eval_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_model(f"epoch-{epoch + 1}")
            
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        # Save final model
        self.save_model("final_model")
        self.logger.info("Training completed!")


def main():
    """Main training function"""
    
    # Training configuration
    config = TrainingConfig(
        model_name="enhanced-supernova",
        batch_size=8,  # Increased batch size
        gradient_accumulation_steps=1,  # No accumulation initially
        learning_rate=5e-5,  # Standard learning rate
        max_epochs=15,
        max_sequence_length=128,  # Start with small sequences
        output_dir="outputs",
        train_data_path="sample_train_data.json",
        mixed_precision=False,  # Keep it simple
        gradient_clipping=1.0,
        warmup_steps=100,
        eval_steps=50,
        save_steps=100,
        logging_steps=5,
        weight_decay=0.01,
        mask_user_tokens=True,  # Focus on response generation
        system_message_weight=0.5,
        response_loss_weight=1.0
    )
    
    # Create trainer and start training
    trainer = SupernovaTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
