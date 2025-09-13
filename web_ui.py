"""
Supernova USLM Web UI - Modern chat interface inspired by DeepSeek
"""

import streamlit as st
import torch
import time
import json
import os
from typing import Optional, List, Dict
import asyncio
from datetime import datetime

# Import our modules
from supernova_model import create_supernova_model
from tokenizer import SupernovaTokenizer
from chat_interface import SupernovaChat
from safety_config import SafetyChecker
from web_search import WebSearch

# Configure Streamlit page
st.set_page_config(
    page_title="Supernova - AI Assistant",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for DeepSeek-inspired design
st.markdown("""
<style>
    .main {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .message-user {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        color: white;
        max-width: 80%;
        margin-left: auto;
        margin-right: 0;
    }
    
    .message-assistant {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        max-width: 80%;
        margin-left: 0;
        margin-right: auto;
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        padding: 12px 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4A90E2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
    }
    
    .supernova-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4A90E2 0%, #50E3C2 50%, #F5A623 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .supernova-subtitle {
        text-align: center;
        color: #8E8E93;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .model-info {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #8E8E93;
        font-style: italic;
    }
    
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #34C759;
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        background-color: #FF3B30;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .quick-actions {
        display: flex;
        gap: 8px;
        margin: 16px 0;
        flex-wrap: wrap;
    }
    
    .quick-action-btn {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 6px 12px;
        color: #ffffff;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .quick-action-btn:hover {
        background: rgba(74, 144, 226, 0.2);
        border-color: #4A90E2;
    }
</style>
""", unsafe_allow_html=True)

class SupernovaWebUI:
    """Web UI for Supernova USLM"""
    
    def __init__(self):
        self.initialize_session_state()
        self.load_model()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        
        if 'chat_instance' not in st.session_state:
            st.session_state.chat_instance = None
        
        if 'model_info' not in st.session_state:
            st.session_state.model_info = {}
        
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_model(self):
        """Load the Supernova model"""
        if not st.session_state.model_loaded:
            with st.spinner("🌟 Initializing Supernova..."):
                try:
                    # Check for trained model
                    model_path = None
                    if os.path.exists("outputs/best_model"):
                        model_path = "outputs/best_model"
                        st.success("✅ Loaded trained Supernova model!")
                    elif os.path.exists("outputs/final_model"):
                        model_path = "outputs/final_model"
                        st.success("✅ Loaded final Supernova model!")
                    else:
                        st.info("ℹ️ No trained model found. Using base model.")
                    
                    # Initialize chat
                    st.session_state.chat_instance = SupernovaChat(
                        model_path=model_path,
                        device="auto"
                    )
                    
                    # Get model info
                    st.session_state.model_info = st.session_state.chat_instance.get_model_info()
                    st.session_state.model_loaded = True
                    
                except Exception as e:
                    st.error(f"❌ Failed to load model: {str(e)}")
                    st.stop()
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="supernova-title">🌟 Supernova</h1>', unsafe_allow_html=True)
        st.markdown('<p class="supernova-subtitle">Ultra-Small Language Model • 25M Parameters • By AlgoRythm Tech</p>', unsafe_allow_html=True)
        
        # Model status
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            if st.session_state.model_loaded:
                st.markdown('<span class="status-indicator status-online"></span>**Model Online**', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-indicator status-offline"></span>**Model Offline**', unsafe_allow_html=True)
        
        with col2:
            params = st.session_state.model_info.get('parameters', 0)
            st.markdown(f"**{params:,}** Parameters")
        
        with col3:
            device = st.session_state.model_info.get('device', 'Unknown')
            st.markdown(f"**Device:** {device}")
        
        with col4:
            vocab_size = st.session_state.model_info.get('vocab_size', 0)
            st.markdown(f"**Vocab:** {vocab_size:,}")
    
    def render_quick_actions(self):
        """Render quick action buttons"""
        st.markdown("### Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💡 Who are you?"):
                st.session_state.messages.append({"role": "user", "content": "Who are you?"})
                self.generate_response()
        
        with col2:
            if st.button("🏢 Tell me about AlgoRythm"):
                st.session_state.messages.append({"role": "user", "content": "Tell me more about AlgoRythm Tech"})
                self.generate_response()
        
        with col3:
            if st.button("🤖 Explain AI"):
                st.session_state.messages.append({"role": "user", "content": "What is artificial intelligence?"})
                self.generate_response()
        
        with col4:
            if st.button("🔬 How do you work?"):
                st.session_state.messages.append({"role": "user", "content": "How do language models like you work?"})
                self.generate_response()
    
    def render_chat_history(self):
        """Render chat messages"""
        if not st.session_state.messages:
            st.markdown("""
            <div class="chat-container">
                <h3>👋 Welcome to Supernova!</h3>
                <p>I'm your AI assistant powered by a 25M parameter language model created by AlgoRythm Tech. 
                I can help you with various tasks including:</p>
                <ul>
                    <li>🤔 Answering questions</li>
                    <li>✍️ Writing and editing</li>
                    <li>📊 Analysis and research</li>
                    <li>💡 Creative tasks</li>
                    <li>🔍 Web search (when enabled)</li>
                </ul>
                <p>How can I assist you today?</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-user">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-assistant">
                    <strong>🌟 Supernova:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    def generate_response(self):
        """Generate AI response"""
        if not st.session_state.messages:
            return
        
        last_message = st.session_state.messages[-1]
        if last_message["role"] != "user":
            return
        
        # Show typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown('<div class="typing-indicator">🌟 Supernova is thinking...</div>', unsafe_allow_html=True)
        
        try:
            # Generate response
            start_time = time.time()
            response = st.session_state.chat_instance.chat(
                last_message["content"],
                add_to_history=False,  # We manage history in session state
                temperature=0.7,
                max_new_tokens=512
            )
            end_time = time.time()
            
            # Clear typing indicator
            typing_placeholder.empty()
            
            # Add response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Show generation time
            st.caption(f"⏱️ Generated in {end_time - start_time:.2f}s")
            
            # Rerun to update chat display
            st.rerun()
            
        except Exception as e:
            typing_placeholder.empty()
            st.error(f"❌ Error generating response: {str(e)}")
    
    def render_sidebar(self):
        """Render sidebar with settings and info"""
        with st.sidebar:
            st.markdown("## 🌟 Supernova Settings")
            
            # Model information
            with st.expander("📊 Model Info", expanded=False):
                if st.session_state.model_info:
                    info = st.session_state.model_info
                    st.json({
                        "Model": info.get('model_name', 'Unknown'),
                        "Parameters": f"{info.get('parameters', 0):,}",
                        "Device": info.get('device', 'Unknown'),
                        "Vocabulary Size": f"{info.get('vocab_size', 0):,}"
                    })
            
            # Generation settings
            with st.expander("⚙️ Generation Settings", expanded=False):
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                max_tokens = st.slider("Max Tokens", 50, 1024, 512, 50)
                top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
                top_k = st.slider("Top-k", 0, 100, 40, 5)
                
                # Store settings in session state
                st.session_state.generation_settings = {
                    'temperature': temperature,
                    'max_new_tokens': max_tokens,
                    'top_p': top_p,
                    'top_k': top_k
                }
            
            # Conversation management
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    if st.session_state.chat_instance:
                        st.session_state.chat_instance.reset_conversation()
                    st.rerun()
            
            with col2:
                if st.button("💾 Save Chat"):
                    self.save_conversation()
            
            # About section
            with st.expander("ℹ️ About Supernova", expanded=False):
                st.markdown("""
                **Supernova USLM** is a 25 million parameter ultra-small language model designed for efficiency and performance.
                
                **Created by:** AlgoRythm Tech  
                **Founder:** Sri Aasrith Souri Kompella  
                **Architecture:** Transformer decoder with modern optimizations
                
                **Features:**
                - Grouped Query Attention (GQA)
                - Rotary Position Embeddings
                - SwiGLU activation functions
                - Sliding window attention
                - Web search capability
                - Safety filtering
                """)
            
            # System status
            st.markdown("---")
            st.markdown("### 🔧 System Status")
            
            if torch.cuda.is_available():
                st.success("✅ CUDA Available")
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    st.info(f"🎮 GPU: {gpu_name}")
            else:
                st.info("💻 Running on CPU")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.memory_reserved() / 1024**3
                st.metric("GPU Memory", f"{memory_used:.1f} / {memory_total:.1f} GB")
    
    def save_conversation(self):
        """Save current conversation"""
        if not st.session_state.messages:
            st.warning("No conversation to save!")
            return
        
        try:
            conversation_data = {
                "conversation_id": st.session_state.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "model_info": st.session_state.model_info
            }
            
            filename = f"conversation_{st.session_state.conversation_id}.json"
            filepath = os.path.join("conversations", filename)
            os.makedirs("conversations", exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"💾 Conversation saved as {filename}")
            
        except Exception as e:
            st.error(f"❌ Failed to save conversation: {str(e)}")
    
    def render_chat_input(self):
        """Render chat input area"""
        # Get generation settings
        settings = getattr(st.session_state, 'generation_settings', {})
        
        # Chat input
        user_input = st.chat_input(
            placeholder="Ask Supernova anything...",
            key="chat_input"
        )
        
        if user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate response
            self.generate_response()
    
    def run(self):
        """Main application runner"""
        # Render main interface
        self.render_header()
        self.render_sidebar()
        
        # Main chat area
        st.markdown("---")
        
        # Show quick actions if no messages
        if not st.session_state.messages:
            self.render_quick_actions()
        
        # Chat history
        self.render_chat_history()
        
        # Chat input
        self.render_chat_input()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #8E8E93; font-size: 0.9rem;'>"
            "🌟 Supernova USLM • Powered by AlgoRythm Tech • "
            f"Made with ❤️ by Sri Aasrith Souri Kompella"
            "</div>",
            unsafe_allow_html=True
        )


def main():
    """Main function to run the Supernova Web UI"""
    try:
        ui = SupernovaWebUI()
        ui.run()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("🔄 Please refresh the page to try again.")


if __name__ == "__main__":
    main()
