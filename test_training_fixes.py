"""
Test script to verify that the training fixes work properly
"""

import torch
import torch.nn as nn
from training import TrainingConfig, SupernovaTrainer
from tokenizer import SupernovaTokenizer, format_training_example

def test_gradient_flow():
    """Test that gradients flow properly through the model"""
    print("🔧 Testing gradient flow...")
    
    # Create a simple config for testing
    config = TrainingConfig(
        model_name="test-model",
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_epochs=1,
        max_sequence_length=128,
        output_dir="test_outputs",
        train_data_path="sample_train_data.json",
        mixed_precision=False,
        gradient_clipping=0.5,
        warmup_steps=10,
        eval_steps=5,
        save_steps=10,
        logging_steps=1,
        weight_decay=0.001,
        num_workers=0,  # Disable multiprocessing for testing
        pin_memory=False,  # Disable for CPU testing
    )
    
    trainer = SupernovaTrainer(config)
    
    # Setup components
    trainer.setup_model_and_tokenizer()
    trainer.setup_datasets()
    trainer.setup_optimizer_and_scheduler()
    
    # Test a single training step manually to check gradients
    for batch in trainer.train_loader:
        trainer.model.train()
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        
        # Forward pass
        trainer.optimizer.zero_grad()
        outputs = trainer.model(**batch)
        loss = outputs['loss']
        
        print(f"✅ Forward pass successful! Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradient norms BEFORE optimizer step
        total_norm = 0.0
        param_count = 0
        nonzero_params = 0
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                if param_norm.item() > 0:
                    nonzero_params += 1
        
        total_norm = total_norm ** 0.5
        print(f"✅ Gradient norm: {total_norm:.6f} (from {param_count} parameters, {nonzero_params} non-zero)")
        
        if total_norm > 0:
            print("🎉 SUCCESS: Gradients are flowing properly!")
        else:
            print("❌ FAILED: Gradients are still zero!")
            
            # Debug: Check a few specific parameters
            print("🔍 Debugging gradient issue:")
            for name, param in list(trainer.model.named_parameters())[:5]:
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    print(f"  {name}: grad_norm = {grad_norm:.8f}")
                else:
                    print(f"  {name}: grad is None")
        
        break  # Only test one batch
    
    return total_norm > 0

def test_label_masking():
    """Test that label masking works correctly"""
    print("\n🔧 Testing label masking...")
    
    tokenizer = SupernovaTokenizer()
    
    # Test conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI."}
    ]
    
    # Format training example
    example = format_training_example(tokenizer, messages, max_length=128)
    
    input_ids = example['input_ids']
    labels = example['labels'] 
    attention_mask = example['attention_mask']
    
    print(f"✅ Input shape: {input_ids.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    print(f"✅ Attention mask shape: {attention_mask.shape}")
    
    # Count unmasked labels
    unmasked_count = (labels != -100).sum().item()
    total_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
    
    print(f"✅ Unmasked labels: {unmasked_count}/{total_tokens} tokens")
    
    if unmasked_count > 0:
        print("🎉 SUCCESS: Labels are properly masked with some unmasked tokens!")
        return True
    else:
        print("❌ FAILED: All labels are masked!")
        return False

def test_loss_computation():
    """Test that loss computation works"""
    print("\n🔧 Testing loss computation...")
    
    from supernova_model import create_supernova_model
    
    # Create a small test model
    model = create_supernova_model(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=512,
        max_position_embeddings=128
    )
    
    model.eval()
    
    # Create test data
    batch_size = 2
    seq_length = 32
    vocab_size = 1000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels = input_ids.clone()
    
    # Mask some labels
    labels[:, :seq_length//2] = -100  # Mask first half
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']
    
    print(f"✅ Loss: {loss.item():.4f}")
    print(f"✅ Logits shape: {logits.shape}")
    
    if loss.item() > 0 and torch.isfinite(loss):
        print("🎉 SUCCESS: Loss computation works properly!")
        return True
    else:
        print("❌ FAILED: Loss is invalid!")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Supernova Training Fixes")
    print("=" * 50)
    
    try:
        # Run tests
        test_results = []
        
        test_results.append(test_label_masking())
        test_results.append(test_loss_computation()) 
        test_results.append(test_gradient_flow())
        
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS:")
        print(f"✅ Passed: {sum(test_results)}/{len(test_results)} tests")
        
        if all(test_results):
            print("🎉 ALL TESTS PASSED! The training fixes should work!")
            print("\n💡 Key fixes applied:")
            print("  • Fixed model initialization (proper transformer init)")
            print("  • Fixed loss computation (removed problematic normalizations)")
            print("  • Fixed learning rate schedule (stable cosine decay with minimum LR)")  
            print("  • Fixed label masking (ensures unmasked tokens for gradients)")
            print("  • Improved hyperparameters (higher LR, better clipping)")
        else:
            print("❌ Some tests failed. Check the issues above.")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
