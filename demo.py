#!/usr/bin/env python3
"""
Demo Script for Supernova USLM - Interactive Chat Interface
Run this script to chat with your trained 25M parameter Supernova model
"""

import os
import sys
import argparse
import torch
import time
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from supernova_model import create_supernova_model, SupernovaConfig
from tokenizer import SupernovaTokenizer
from chat_interface import SupernovaChat, interactive_chat
from training import create_sample_training_data, SupernovaTrainer, TrainingConfig


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════╗
║                                                      ║
║              🌟 SUPERNOVA USLM 🌟                    ║
║         Ultra-Small Language Model (25M params)     ║
║                                                      ║
║    A compact, efficient chat model designed to      ║
║    deliver helpful responses with minimal compute    ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def test_model_basic():
    """Test basic model functionality"""
    print("\n🧪 Testing basic model functionality...")
    
    try:
        # Create model and tokenizer
        model = create_supernova_model()
        tokenizer = SupernovaTokenizer()
        
        # Test forward pass
        test_text = "Hello, how are you?"
        input_ids = torch.tensor([tokenizer.encode(test_text)], dtype=torch.long)
        
        print(f"✅ Model created successfully: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"✅ Tokenizer loaded: {len(tokenizer)} vocabulary size")
        print(f"✅ Test input: '{test_text}' -> {len(input_ids[0])} tokens")
        
        # Test inference
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"✅ Forward pass successful: output shape {outputs['logits'].shape}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error during basic test: {e}")
        return None, None


def run_training_demo(quick_demo: bool = True):
    """Run a quick training demonstration"""
    print("\n🚀 Running training demonstration...")
    
    try:
        # Create sample data
        print("📝 Creating sample training data...")
        data_path = create_sample_training_data("demo_train_data.json", num_samples=50 if quick_demo else 200)
        
        # Setup training config for demo
        config = TrainingConfig(
            model_name="supernova-demo",
            batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=3e-5,
            max_epochs=1 if quick_demo else 2,
            max_steps=10 if quick_demo else None,
            max_sequence_length=256,
            output_dir="demo_outputs",
            train_data_path=data_path,
            mixed_precision=False,  # Disable for demo compatibility
            eval_steps=5,
            save_steps=20,
            logging_steps=2,
            num_workers=0  # Avoid multiprocessing issues on Windows
        )
        
        print(f"⚙️  Training config: {config.max_epochs} epochs, {config.max_steps or 'unlimited'} steps")
        
        # Run training
        trainer = SupernovaTrainer(config)
        trainer.train()
        
        print("✅ Training demonstration completed!")
        return os.path.join(config.output_dir, "final_model")
        
    except Exception as e:
        print(f"❌ Training demo error: {e}")
        return None


def chat_demo(model_path: Optional[str] = None):
    """Run interactive chat demonstration"""
    print("\n💬 Starting chat demonstration...")
    
    try:
        # Initialize chat interface
        if model_path and os.path.exists(model_path):
            print(f"📂 Loading trained model from: {model_path}")
            chat = SupernovaChat(model_path=model_path)
        else:
            print("🆕 Using fresh untrained model (responses may be incoherent)")
            model = create_supernova_model()
            tokenizer = SupernovaTokenizer()
            chat = SupernovaChat(model=model, tokenizer=tokenizer)
        
        print("✅ Chat interface ready!")
        
        # Test a few sample interactions
        sample_questions = [
            "Hello! What are you?",
            "What is machine learning?",
            "Can you help me with coding?",
            "Tell me a fun fact."
        ]
        
        print("\n🤖 Testing sample interactions:")
        print("=" * 50)
        
        for question in sample_questions[:2]:  # Test first 2 questions
            print(f"\n👤 User: {question}")
            print("🤖 Supernova: ", end="", flush=True)
            
            start_time = time.time()
            try:
                response = chat.chat(question, temperature=0.8, max_new_tokens=100)
                end_time = time.time()
                
                print(response)
                print(f"⏱️  ({end_time - start_time:.2f}s)")
                
            except Exception as e:
                print(f"Error generating response: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 Sample interactions complete!")
        
        # Ask if user wants full interactive mode
        print("\nWould you like to start full interactive chat mode? (y/n): ", end="")
        if input().lower().strip() in ['y', 'yes']:
            print("\n🚀 Starting interactive mode...")
            interactive_chat(model_path=model_path)
        else:
            print("👋 Demo completed!")
        
    except Exception as e:
        print(f"❌ Chat demo error: {e}")


def setup_demo_environment():
    """Setup and verify demo environment"""
    print("\n🔧 Setting up demo environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    try:
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Running on CPU")
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        return False
    
    # Check required files
    required_files = [
        "supernova_model.py",
        "tokenizer.py", 
        "chat_interface.py",
        "training.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing required files: {missing_files}")
        return False
    
    print("✅ Environment setup complete!")
    return True


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Supernova USLM Demo")
    parser.add_argument("--mode", choices=["test", "train", "chat", "full"], default="full",
                       help="Demo mode: test (basic), train (training demo), chat (chat only), full (all)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model directory")
    parser.add_argument("--quick", action="store_true",
                       help="Quick demo mode (faster but less comprehensive)")
    parser.add_argument("--skip-train", action="store_true",
                       help="Skip training demonstration")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup environment
    if not setup_demo_environment():
        print("❌ Environment setup failed. Please check dependencies.")
        return
    
    # Run selected demo mode
    trained_model_path = args.model_path
    
    if args.mode in ["test", "full"]:
        print("\n" + "="*60)
        print("🧪 BASIC MODEL TEST")
        print("="*60)
        model, tokenizer = test_model_basic()
        if not model:
            print("❌ Basic test failed. Cannot continue.")
            return
    
    if args.mode in ["train", "full"] and not args.skip_train:
        print("\n" + "="*60)
        print("🚀 TRAINING DEMONSTRATION")
        print("="*60)
        trained_model_path = run_training_demo(quick_demo=args.quick)
    
    if args.mode in ["chat", "full"]:
        print("\n" + "="*60)
        print("💬 CHAT DEMONSTRATION") 
        print("="*60)
        chat_demo(trained_model_path)
    
    print("\n🎉 Demo completed! Thank you for trying Supernova USLM!")
    print("\nTo run individual components:")
    print("  python demo.py --mode test     # Test basic functionality")
    print("  python demo.py --mode train    # Training demonstration")  
    print("  python demo.py --mode chat     # Chat interface only")
    print("  python chat_interface.py      # Direct interactive chat")
    print("  python training.py            # Full training pipeline")


if __name__ == "__main__":
    main()
