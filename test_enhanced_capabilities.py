#!/usr/bin/env python3
"""
Test Script for Enhanced Supernova Capabilities
Tests all advanced tools and integrations
"""

import sys
import traceback
from datetime import datetime

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    try:
        import torch
        import transformers
        import numpy as np
        print("✅ Core ML libraries imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import core libraries: {e}")
        return False

def test_enhanced_imports():
    """Test enhanced capability imports"""
    print("🔍 Testing enhanced imports...")
    try:
        import sympy
        import matplotlib.pyplot as plt
        import requests
        import streamlit
        print("✅ Enhanced libraries imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import enhanced libraries: {e}")
        print("💡 Run: pip install sympy matplotlib requests streamlit")
        return False

def test_model_loading():
    """Test model and tokenizer loading"""
    print("🔍 Testing model loading...")
    try:
        from supernova_model import create_supernova_model
        from tokenizer import SupernovaTokenizer
        
        model = create_supernova_model()
        tokenizer = SupernovaTokenizer()
        
        print("✅ Model and tokenizer loaded successfully")
        print(f"   📊 Model parameters: {model.num_parameters():,}")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_advanced_tools():
    """Test advanced tools system"""
    print("🔍 Testing advanced tools...")
    try:
        from advanced_tools import AdvancedToolSystem
        
        tools = AdvancedToolSystem()
        
        # Test math solving
        print("  🧮 Testing math solving...")
        result = tools.solve_math("solve x^2 + 5x + 6 = 0")
        if result['success']:
            print(f"    ✅ Math result: {result['solution']}")
        else:
            print(f"    ⚠️ Math solving failed: {result.get('error', 'Unknown error')}")
        
        # Test code execution
        print("  💻 Testing code execution...")
        code = "print('Hello from Enhanced Supernova!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')"
        result = tools.execute_code(code)
        if result['success']:
            print(f"    ✅ Code executed successfully")
            print(f"    📤 Output: {result['output'].strip()}")
        else:
            print(f"    ⚠️ Code execution failed: {result.get('error', 'Unknown error')}")
        
        # Test capabilities
        caps = tools.get_capabilities()
        print("  🛠️ Available tools:")
        for tool, available in tools.tools.items():
            status = "✅" if available else "❌"
            print(f"    {status} {tool.upper()}")
        
        tools.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Advanced tools test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_chat():
    """Test enhanced chat system"""
    print("🔍 Testing enhanced chat system...")
    try:
        from advanced_tools import create_enhanced_chat_system
        
        EnhancedChat, tools = create_enhanced_chat_system()
        chat = EnhancedChat()
        
        # Test with basic question
        print("  💬 Testing basic conversation...")
        response = chat.chat("Hello, who are you?", enhance_with_tools=False)
        print(f"    📝 Response length: {len(response)} characters")
        
        print("✅ Enhanced chat system working")
        tools.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Enhanced chat test failed: {e}")
        traceback.print_exc()
        return False

def test_safety_and_web():
    """Test safety and web search components"""
    print("🔍 Testing safety and web components...")
    try:
        from safety_config import SafetyChecker
        from web_search import WebSearch
        
        # Test safety checker
        safety = SafetyChecker()
        print("  🛡️ Safety checker initialized")
        
        # Test web search (if API key available)
        try:
            web = WebSearch()
            print("  🔍 Web search initialized")
        except Exception:
            print("  ⚠️ Web search needs API key (optional)")
        
        print("✅ Safety and web components loaded")
        return True
        
    except Exception as e:
        print(f"❌ Safety/web test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Enhanced Supernova Capabilities Test")
    print("=" * 50)
    print(f"📅 Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Enhanced Imports", test_enhanced_imports), 
        ("Model Loading", test_model_loading),
        ("Advanced Tools", test_advanced_tools),
        ("Enhanced Chat", test_enhanced_chat),
        ("Safety & Web", test_safety_and_web)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'─' * 30}")
        print(f"🧪 Running: {test_name}")
        print('─' * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📊 TEST SUMMARY")
    print('=' * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<10} {test_name}")
    
    print(f"\n🏆 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Enhanced Supernova is ready!")
        print("🚀 You can now run:")
        print("   • python run_enhanced_webui.py  (Enhanced Web UI)")
        print("   • python run_webui.py          (Basic Web UI)")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check the errors above.")
        print("💡 Make sure all requirements are installed: pip install -r requirements.txt")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
