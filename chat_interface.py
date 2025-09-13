"""
Chat Interface for Supernova USLM - Enhanced conversation handling and generation
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
import json
import os
import logging
from transformers import StoppingCriteria, StoppingCriteriaList

from supernova_model import SupernovaForCausalLM, SupernovaConfig
from tokenizer import SupernovaTokenizer
from safety_config import SafetyChecker
from web_search import WebSearch, SearchEnhancedResponse


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 512
    max_new_tokens: Optional[int] = None
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    output_scores: bool = False
    return_dict_in_generate: bool = True


class ChatStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for chat responses"""
    
    def __init__(self, stop_strings: List[str], tokenizer: SupernovaTokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.stop_ids = []
        
        # Convert stop strings to token IDs
        for stop_str in stop_strings:
            stop_tokens = tokenizer.encode(stop_str, add_special_tokens=False)
            self.stop_ids.extend(stop_tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any stop token appears in the last few tokens
        for stop_id in self.stop_ids:
            if stop_id in input_ids[0, -10:]:  # Check last 10 tokens
                return True
        
        # Check if decoded text contains stop strings
        try:
            last_tokens = input_ids[0, -20:]  # Check last 20 tokens
            decoded = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
            for stop_str in self.stop_strings:
                if stop_str.lower() in decoded.lower():
                    return True
        except:
            pass
        
        return False


class SupernovaChat:
    """Main chat interface for Supernova USLM"""
    
    def __init__(
        self,
        model: Optional[SupernovaForCausalLM] = None,
        tokenizer: Optional[SupernovaTokenizer] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        generation_config: Optional[GenerationConfig] = None
    ):
        self.device = self._setup_device(device)
        
        # Load or use provided model and tokenizer
        if model is None or tokenizer is None:
            self.model, self.tokenizer = self._load_model_and_tokenizer(model_path)
        else:
            self.model = model
            self.tokenizer = tokenizer
        
        self.model.to(self.device)
        self.model.eval()
        
        # Generation configuration
        self.generation_config = generation_config or GenerationConfig()
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Chat history
        self.chat_history: List[Dict[str, str]] = []
        
        # Initialize safety checker
        self.safety_checker = SafetyChecker()
        
        # Initialize web search (optional)
        try:
            self.web_search = SearchEnhancedResponse()
            self.search_enabled = True
        except:
            self.web_search = None
            self.search_enabled = False
        
        # System prompt
        self.system_prompt = (
            "You are Supernova, a helpful, harmless, and honest AI assistant created by AlgoRythm Tech. "
            "You aim to be helpful, accurate, and engaging while being concise. "
            "You can assist with various tasks including answering questions, writing, analysis, and creative tasks. "
            "When providing information, prioritize accuracy and helpfulness while following all safety guidelines."
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"SupernovaChat initialized on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
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
    
    def _load_model_and_tokenizer(self, model_path: Optional[str]) -> Tuple[SupernovaForCausalLM, SupernovaTokenizer]:
        """Load model and tokenizer"""
        if model_path and os.path.exists(model_path):
            # Load from saved checkpoint
            self.logger.info(f"Loading model from {model_path}")
            config = SupernovaConfig()  # Default config
            model = SupernovaForCausalLM(config)
            model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location="cpu"))
            tokenizer = SupernovaTokenizer(os.path.join(model_path, "tokenizer"))
        else:
            # Create new model
            self.logger.info("Creating new Supernova model")
            from supernova_model import create_supernova_model
            model = create_supernova_model()
            tokenizer = SupernovaTokenizer()
        
        return model, tokenizer
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt for conversations"""
        self.system_prompt = prompt
        self.logger.info("System prompt updated")
    
    def reset_conversation(self):
        """Reset chat history"""
        self.chat_history.clear()
        self.logger.info("Conversation reset")
    
    def add_message(self, role: str, content: str):
        """Add message to chat history"""
        self.chat_history.append({"role": role, "content": content})
    
    def prepare_messages_for_generation(self, user_input: str, include_system: bool = True) -> List[Dict[str, str]]:
        """Prepare messages for generation"""
        messages = []
        
        # Add system prompt if requested
        if include_system and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add chat history
        messages.extend(self.chat_history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def generate_response(
        self,
        user_input: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, torch.Tensor]:
        """Generate response to user input"""
        
        # Update generation config with provided parameters
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.generation_config.max_new_tokens or 256,
            temperature=temperature or self.generation_config.temperature,
            top_p=top_p or self.generation_config.top_p,
            top_k=top_k or self.generation_config.top_k,
            repetition_penalty=repetition_penalty or self.generation_config.repetition_penalty,
            do_sample=do_sample if do_sample is not None else self.generation_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **kwargs
        )
        
        # Prepare conversation
        messages = self.prepare_messages_for_generation(user_input)
        formatted_prompt = self.tokenizer.format_chat_prompt(messages, add_generation_prompt=True)
        
        # Tokenize
        inputs = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        input_ids = torch.tensor([inputs], dtype=torch.long, device=self.device)
        
        # Setup stopping criteria
        stop_strings = [
            self.tokenizer.special_tokens['user_token'],
            self.tokenizer.special_tokens['system_token'],
            "\n<|user|>",
            "\n<|system|>",
            "<|user|>",
            "<|system|>"
        ]
        
        stopping_criteria = StoppingCriteriaList([
            ChatStoppingCriteria(stop_strings, self.tokenizer)
        ])
        
        start_time = time.time()
        
        if stream:
            return self._generate_streaming(input_ids, gen_config, stopping_criteria)
        else:
            return self._generate_complete(input_ids, gen_config, stopping_criteria, start_time)
    
    def _generate_complete(
        self, 
        input_ids: torch.Tensor, 
        gen_config: GenerationConfig, 
        stopping_criteria: StoppingCriteriaList,
        start_time: float
    ) -> str:
        """Generate complete response"""
        
        with torch.no_grad():
            # Generate
            generated_ids = self.advanced_generate(
                input_ids=input_ids,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_k=gen_config.top_k,
                top_p=gen_config.top_p,
                repetition_penalty=gen_config.repetition_penalty,
                do_sample=gen_config.do_sample,
                eos_token_id=gen_config.eos_token_id,
                pad_token_id=gen_config.pad_token_id,
                stopping_criteria=stopping_criteria
            )
            
            # Decode response (only new tokens)
            new_tokens = generated_ids[0, input_ids.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = self._clean_response(response)
            
            generation_time = time.time() - start_time
            tokens_generated = len(new_tokens)
            
            self.logger.info(
                f"Generated {tokens_generated} tokens in {generation_time:.2f}s "
                f"({tokens_generated/generation_time:.1f} tok/s)"
            )
            
            return response
    
    def _generate_streaming(
        self, 
        input_ids: torch.Tensor, 
        gen_config: GenerationConfig, 
        stopping_criteria: StoppingCriteriaList
    ):
        """Generate streaming response (returns generator)"""
        
        current_ids = input_ids.clone()
        generated_text = ""
        
        with torch.no_grad():
            for _ in range(gen_config.max_new_tokens or 256):
                # Generate next token
                outputs = self.model(current_ids, use_cache=True)
                logits = outputs['logits'][:, -1, :]
                
                # Apply sampling
                next_token_id = self._sample_next_token(
                    logits, 
                    gen_config.temperature,
                    gen_config.top_k,
                    gen_config.top_p,
                    gen_config.do_sample
                )
                
                # Add to sequence
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)
                
                # Decode new token
                new_token_text = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
                generated_text += new_token_text
                
                yield new_token_text
                
                # Check stopping criteria
                if stopping_criteria(current_ids, logits):
                    break
                
                if next_token_id.item() == gen_config.eos_token_id:
                    break
    
    def advanced_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ) -> torch.Tensor:
        """Advanced generation with better sampling and control"""
        
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]
        device = input_ids.device
        
        # Initialize
        generated = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_new_tokens):
            # Prepare input (use only last token if we have past_key_values)
            model_input = generated if past_key_values is None else generated[:, -1:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=model_input,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs['logits'][:, -1, :]
                past_key_values = outputs.get('past_key_values')
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
                
                # Sample next token
                if do_sample:
                    next_tokens = self._sample_next_token(logits, temperature, top_k, top_p, do_sample=True)
                else:
                    next_tokens = torch.argmax(logits, dim=-1)
                
                # Update generated sequence
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
                
                # Check for EOS tokens
                if eos_token_id is not None:
                    finished = finished | (next_tokens == eos_token_id)
                
                # Check stopping criteria
                if stopping_criteria is not None:
                    should_stop = any(criteria(generated, logits) for criteria in stopping_criteria)
                    if should_stop:
                        break
                
                # Check if all sequences are finished
                if finished.all():
                    break
        
        return generated
    
    def _sample_next_token(
        self, 
        logits: torch.Tensor, 
        temperature: float, 
        top_k: int, 
        top_p: float, 
        do_sample: bool = True
    ) -> torch.Tensor:
        """Sample next token with various strategies"""
        
        if not do_sample:
            return torch.argmax(logits, dim=-1)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_tokens
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated_ids: torch.Tensor, 
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        
        for i in range(logits.shape[0]):
            for token_id in set(generated_ids[i].tolist()):
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= penalty
                else:
                    logits[i, token_id] *= penalty
        
        return logits
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response"""
        
        # Remove special tokens that might have leaked through
        for token in self.tokenizer.special_tokens.values():
            response = response.replace(token, "")
        
        # Remove common artifacts
        response = response.replace("<|user|>", "").replace("<|assistant|>", "").replace("<|system|>", "")
        
        # Strip whitespace and clean up
        response = response.strip()
        
        # Remove incomplete sentences at the end
        if response and response[-1] not in '.!?':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response
    
    def chat(self, user_input: str, add_to_history: bool = True, enable_search: bool = True, **generation_kwargs) -> str:
        """Enhanced chat interface with safety checks and web search"""
        
        # Check input safety
        is_safe, refusal_message = self.safety_checker.check_input(user_input)
        if not is_safe:
            return refusal_message
        
        # Check for company info queries first
        company_response = self.safety_checker.format_company_info_response(user_input)
        if company_response:
            if add_to_history:
                self.add_message("user", user_input)
                self.add_message("assistant", company_response)
            return company_response
        
        # Generate AI response
        response = self.generate_response(user_input, **generation_kwargs)
        
        # Apply safety checks to output
        is_output_safe, safe_response = self.safety_checker.check_output(response)
        response = safe_response
        
        # Enhance with web search if enabled and available
        if enable_search and self.search_enabled:
            try:
                response = self.web_search.enhance_response(
                    user_input, 
                    response, 
                    should_search=True,
                    num_results=3
                )
            except Exception as e:
                self.logger.warning(f"Web search failed: {e}")
        
        if add_to_history:
            # Add to chat history
            self.add_message("user", user_input)
            self.add_message("assistant", response)
        
        return response
    
    def save_conversation(self, filepath: str):
        """Save conversation to file"""
        conversation_data = {
            "system_prompt": self.system_prompt,
            "chat_history": self.chat_history,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str):
        """Load conversation from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        
        self.system_prompt = conversation_data.get("system_prompt", self.system_prompt)
        self.chat_history = conversation_data.get("chat_history", [])
        
        self.logger.info(f"Conversation loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": "Supernova USLM",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "vocab_size": len(self.tokenizer),
            "config": self.model.config.__dict__ if hasattr(self.model, 'config') else {},
            "generation_config": self.generation_config.__dict__
        }


def interactive_chat(model_path: Optional[str] = None, device: str = "auto"):
    """Interactive chat session"""
    
    print("🌟 Supernova USLM Chat Interface")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'reset' to clear conversation history")
    print("Type 'system <prompt>' to change system prompt")
    print("Type 'save <filename>' to save conversation")
    print("Type 'load <filename>' to load conversation")
    print("-" * 50)
    
    # Initialize chat
    try:
        chat = SupernovaChat(model_path=model_path, device=device)
        print(f"Model loaded successfully! Parameters: {sum(p.numel() for p in chat.model.parameters()):,}")
        print(f"Device: {chat.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'reset':
                chat.reset_conversation()
                print("🔄 Conversation history cleared!")
                continue
            
            elif user_input.lower().startswith('system '):
                new_prompt = user_input[7:]
                chat.set_system_prompt(new_prompt)
                print(f"🔧 System prompt updated!")
                continue
            
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip() or "conversation.json"
                try:
                    chat.save_conversation(filename)
                    print(f"💾 Conversation saved to {filename}")
                except Exception as e:
                    print(f"❌ Error saving: {e}")
                continue
            
            elif user_input.lower().startswith('load '):
                filename = user_input[5:].strip()
                try:
                    chat.load_conversation(filename)
                    print(f"📂 Conversation loaded from {filename}")
                except Exception as e:
                    print(f"❌ Error loading: {e}")
                continue
            
            # Generate response
            print("🤖 Supernova: ", end="", flush=True)
            
            start_time = time.time()
            response = chat.chat(user_input)
            end_time = time.time()
            
            print(response)
            print(f"\n⏱️  Generated in {end_time - start_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    # Start interactive chat
    interactive_chat()
