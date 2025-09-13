"""
Enhanced Supernova USLM Web UI with Advanced Tools
Includes math solving, code execution, plotting, and more capabilities
"""

import streamlit as st
import torch
import time
import json
import os
import re
from typing import Optional, List, Dict
import asyncio
from datetime import datetime
import base64

# Import our modules
from supernova_model import create_supernova_model
from tokenizer import SupernovaTokenizer
from chat_interface import SupernovaChat
from safety_config import SafetyChecker
from web_search import WebSearch
from advanced_tools import AdvancedToolSystem, create_enhanced_chat_system

# Configure Streamlit page
st.set_page_config(
    page_title="Supernova - Enhanced AI Assistant",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before but with additions for tool outputs)
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
    
    .tool-output {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-family: 'Courier New', monospace;
    }
    
    .math-solution {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #FFC107;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    
    .code-output {
        background: rgba(156, 39, 176, 0.1);
        border-left: 4px solid #9C27B0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
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
    
    .capabilities-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 16px 0;
    }
    
    .capability-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .capability-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedSupernovaWebUI:
    """Enhanced Web UI for Supernova USLM with advanced tools"""
    
    def __init__(self):
        self.initialize_session_state()
        self.load_enhanced_model()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        
        if 'chat_instance' not in st.session_state:
            st.session_state.chat_instance = None
        
        if 'tools_instance' not in st.session_state:
            st.session_state.tools_instance = None
        
        if 'model_info' not in st.session_state:
            st.session_state.model_info = {}
        
        if 'tools_enabled' not in st.session_state:
            st.session_state.tools_enabled = True
        
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_enhanced_model(self):
        """Load the enhanced Supernova model with tools"""
        if not st.session_state.model_loaded:
            with st.spinner("🛠️ Initializing Enhanced Supernova..."):
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
                    
                    # Initialize enhanced chat with tools
                    EnhancedChat, tools = create_enhanced_chat_system()
                    st.session_state.chat_instance = EnhancedChat(
                        model_path=model_path,
                        device="auto"
                    )
                    st.session_state.tools_instance = tools
                    
                    # Get model info
                    st.session_state.model_info = st.session_state.chat_instance.get_capabilities()
                    st.session_state.model_loaded = True
                    
                except Exception as e:
                    st.error(f"❌ Failed to load enhanced model: {str(e)}")
                    st.stop()
    
    def render_header(self):
        """Render the main header with enhanced capabilities"""
        st.markdown('<h1 class="supernova-title">🌟 Supernova Enhanced</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #8E8E93; font-size: 1.1rem;">Ultra-Small Language Model with Advanced Tools • Math • Code • Plots • Web Search</p>', unsafe_allow_html=True)
        
        # Enhanced status display
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            if st.session_state.model_loaded:
                st.markdown('✅ **Model Online**')
            else:
                st.markdown('❌ **Model Offline**')
        
        with col2:
            if st.session_state.tools_instance:
                tools_count = sum(st.session_state.tools_instance.tools.values())
                st.markdown(f'🛠️ **{tools_count}/5 Tools**')
        
        with col3:
            device = st.session_state.model_info.get('model', {}).get('device', 'Unknown')
            st.markdown(f"**Device:** {device}")
        
        with col4:
            if st.session_state.tools_enabled:
                st.markdown('🚀 **Enhanced Mode**')
            else:
                st.markdown('🔧 **Basic Mode**')
    
    def render_capabilities(self):
        """Render enhanced capabilities display"""
        if st.session_state.tools_instance:
            st.markdown("### 🛠️ Available Capabilities")
            
            # Get tool status
            tools = st.session_state.tools_instance.tools
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                status = "🧮 ✅" if tools['math'] else "🧮 ❌"
                st.markdown(f"**{status}**<br>Math Solving", unsafe_allow_html=True)
                if tools['math']:
                    st.caption("SymPy + Wolfram")
            
            with col2:
                status = "💻 ✅" if tools['code'] else "💻 ❌"
                st.markdown(f"**{status}**<br>Code Execution", unsafe_allow_html=True)
                if tools['code']:
                    st.caption("Python + Safety")
            
            with col3:
                status = "📊 ✅" if tools['plotting'] else "📊 ❌"
                st.markdown(f"**{status}**<br>Data Plotting", unsafe_allow_html=True)
                if tools['plotting']:
                    st.caption("Matplotlib")
            
            with col4:
                status = "🔍 ✅" if tools['web'] else "🔍 ❌"
                st.markdown(f"**{status}**<br>Web Search", unsafe_allow_html=True)
                if tools['web']:
                    st.caption("Serper API")
            
            with col5:
                status = "🔬 ✅" if tools['wolfram'] else "🔬 ❌"
                st.markdown(f"**{status}**<br>Wolfram Alpha", unsafe_allow_html=True)
                if tools['wolfram']:
                    st.caption("Advanced Math")
                else:
                    st.caption("API Key Needed")
    
    def render_enhanced_quick_actions(self):
        """Render enhanced quick action buttons"""
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧮 Solve: x² + 5x + 6 = 0"):
                st.session_state.messages.append({"role": "user", "content": "Solve the equation x² + 5x + 6 = 0"})
                self.generate_enhanced_response()
            
            if st.button("💻 Code: Print Fibonacci"):
                st.session_state.messages.append({"role": "user", "content": "Write Python code to print the first 10 Fibonacci numbers"})
                self.generate_enhanced_response()
            
            if st.button("📊 Plot: Sine Wave"):
                st.session_state.messages.append({"role": "user", "content": "Create a plot of sine wave from 0 to 2π"})
                self.generate_enhanced_response()
        
        with col2:
            if st.button("💡 Who are you?"):
                st.session_state.messages.append({"role": "user", "content": "Who are you and what can you do?"})
                self.generate_enhanced_response()
            
            if st.button("🏢 About AlgoRythm"):
                st.session_state.messages.append({"role": "user", "content": "Tell me about AlgoRythm Tech"})
                self.generate_enhanced_response()
            
            if st.button("🔬 What is calculus?"):
                st.session_state.messages.append({"role": "user", "content": "What is calculus and show me the derivative of x²"})
                self.generate_enhanced_response()
    
    def render_chat_history(self):
        """Render enhanced chat messages with tool outputs"""
        if not st.session_state.messages:
            st.markdown("""
            <div class="chat-container">
                <h3>👋 Welcome to Enhanced Supernova!</h3>
                <p>I'm your AI assistant with advanced capabilities:</p>
                <div class="capabilities-grid">
                    <div class="capability-card">
                        <div class="capability-icon">🧮</div>
                        <strong>Math Solving</strong><br>
                        <small>Equations, calculus, algebra</small>
                    </div>
                    <div class="capability-card">
                        <div class="capability-icon">💻</div>
                        <strong>Code Execution</strong><br>
                        <small>Python with safety checks</small>
                    </div>
                    <div class="capability-card">
                        <div class="capability-icon">📊</div>
                        <strong>Data Plotting</strong><br>
                        <small>Charts, graphs, visualizations</small>
                    </div>
                    <div class="capability-card">
                        <div class="capability-icon">🔍</div>
                        <strong>Web Search</strong><br>
                        <small>Real-time information</small>
                    </div>
                </div>
                <p>Try asking me to solve math problems, write code, create plots, or answer questions!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat messages with enhanced formatting
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-user">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Parse enhanced response for special sections
                content = message["content"]
                
                # Check for math solutions
                if "🧮 **Mathematical Solution:**" in content:
                    parts = content.split("🧮 **Mathematical Solution:**")
                    main_content = parts[0]
                    math_content = parts[1] if len(parts) > 1 else ""
                    
                    st.markdown(f"""
                    <div class="message-assistant">
                        <strong>🌟 Supernova:</strong><br>
                        {main_content}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if math_content:
                        st.markdown(f"""
                        <div class="math-solution">
                            <strong>🧮 Mathematical Solution:</strong><br>
                            {math_content}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Check for code execution
                elif "💻 **Code Execution Result:**" in content:
                    parts = content.split("💻 **Code Execution Result:**")
                    main_content = parts[0]
                    code_content = parts[1] if len(parts) > 1 else ""
                    
                    st.markdown(f"""
                    <div class="message-assistant">
                        <strong>🌟 Supernova:</strong><br>
                        {main_content}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if code_content:
                        st.markdown(f"""
                        <div class="code-output">
                            <strong>💻 Code Execution Result:</strong><br>
                            {code_content}
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    # Regular message
                    st.markdown(f"""
                    <div class="message-assistant">
                        <strong>🌟 Supernova:</strong><br>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
    
    def generate_enhanced_response(self):
        """Generate AI response with enhanced tools"""
        if not st.session_state.messages:
            return
        
        last_message = st.session_state.messages[-1]
        if last_message["role"] != "user":
            return
        
        # Show typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown('<div style="color: #8E8E93; font-style: italic;">🌟 Supernova is thinking with advanced tools...</div>', unsafe_allow_html=True)
        
        try:
            # Generate enhanced response
            start_time = time.time()
            
            # Get generation settings
            settings = getattr(st.session_state, 'generation_settings', {})
            
            response = st.session_state.chat_instance.chat(
                last_message["content"],
                add_to_history=False,
                enhance_with_tools=st.session_state.tools_enabled,
                temperature=settings.get('temperature', 0.7),
                max_new_tokens=settings.get('max_new_tokens', 512),
                top_p=settings.get('top_p', 0.9),
                top_k=settings.get('top_k', 40)
            )
            
            end_time = time.time()
            
            # Clear typing indicator
            typing_placeholder.empty()
            
            # Add response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Show generation time and tool usage
            tool_usage = "with advanced tools" if st.session_state.tools_enabled else "basic mode"
            st.caption(f"⏱️ Generated in {end_time - start_time:.2f}s ({tool_usage})")
            
            # Rerun to update chat display
            st.rerun()
            
        except Exception as e:
            typing_placeholder.empty()
            st.error(f"❌ Error generating enhanced response: {str(e)}")
    
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with tool controls"""
        with st.sidebar:
            st.markdown("## 🌟 Supernova Enhanced")
            
            # Tools toggle
            st.session_state.tools_enabled = st.toggle(
                "🛠️ Advanced Tools", 
                value=st.session_state.tools_enabled,
                help="Enable math solving, code execution, and plotting capabilities"
            )
            
            # Tool status
            if st.session_state.tools_instance:
                with st.expander("🔧 Tool Status", expanded=False):
                    tools = st.session_state.tools_instance.tools
                    for tool, available in tools.items():
                        status = "✅" if available else "❌"
                        st.write(f"{status} {tool.upper()}")
                
                # Advanced tool settings
                if st.session_state.tools_enabled:
                    with st.expander("⚙️ Tool Settings", expanded=False):
                        # Wolfram API key input
                        if not tools['wolfram']:
                            wolfram_key = st.text_input(
                                "Wolfram Alpha API Key",
                                type="password",
                                help="Enter your Wolfram Alpha API key for advanced math solving"
                            )
                            if wolfram_key:
                                st.session_state.tools_instance.wolfram_api_key = wolfram_key
                                st.session_state.tools_instance.tools['wolfram'] = True
                                st.success("✅ Wolfram Alpha enabled!")
                        else:
                            st.success("✅ Wolfram Alpha ready!")
                        
                        # Code execution settings
                        st.write("**Code Execution:**")
                        st.write("• Timeout: 10 seconds")
                        st.write("• Language: Python only")
                        st.write("• Safety checks enabled")
            
            # Model information
            with st.expander("📊 Enhanced Model Info", expanded=False):
                if st.session_state.model_info:
                    model_info = st.session_state.model_info.get('model', {})
                    tool_info = st.session_state.model_info.get('advanced_tools', {})
                    
                    st.write("**Base Model:**")
                    st.json({
                        "Parameters": f"{model_info.get('parameters', 0):,}",
                        "Device": model_info.get('device', 'Unknown'),
                        "Vocab Size": f"{model_info.get('vocab_size', 0):,}"
                    })
                    
                    st.write("**Enhanced Capabilities:**")
                    st.json({
                        "Math Solving": tool_info.get('math_solving', {}).get('sympy', False),
                        "Wolfram Alpha": tool_info.get('math_solving', {}).get('wolfram_alpha', False),
                        "Code Execution": tool_info.get('code_execution', {}).get('python', False),
                        "Plotting": tool_info.get('plotting', {}).get('available', False)
                    })
            
            # Generation settings (same as before)
            with st.expander("⚙️ Generation Settings", expanded=False):
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                max_tokens = st.slider("Max Tokens", 50, 1024, 512, 50)
                top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
                top_k = st.slider("Top-k", 0, 100, 40, 5)
                
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
    
    def save_conversation(self):
        """Save current conversation with enhanced metadata"""
        if not st.session_state.messages:
            st.warning("No conversation to save!")
            return
        
        try:
            conversation_data = {
                "conversation_id": st.session_state.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "model_info": st.session_state.model_info,
                "tools_enabled": st.session_state.tools_enabled,
                "tool_status": st.session_state.tools_instance.tools if st.session_state.tools_instance else {}
            }
            
            filename = f"enhanced_conversation_{st.session_state.conversation_id}.json"
            filepath = os.path.join("conversations", filename)
            os.makedirs("conversations", exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"💾 Enhanced conversation saved as {filename}")
            
        except Exception as e:
            st.error(f"❌ Failed to save conversation: {str(e)}")
    
    def render_chat_input(self):
        """Render enhanced chat input area"""
        # Examples for users
        st.markdown("### 💭 Try asking:")
        st.markdown("""
        - **Math**: "Solve x² + 5x + 6 = 0" or "What's the derivative of x³?"
        - **Code**: "Write Python code to calculate factorial" 
        - **Plot**: "Create a bar chart of [1,3,2,5,4]"
        - **General**: "Who are you?" or "Explain machine learning"
        """)
        
        # Chat input
        user_input = st.chat_input(
            placeholder="Ask Supernova anything - math, code, plots, or general questions...",
            key="enhanced_chat_input"
        )
        
        if user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate enhanced response
            self.generate_enhanced_response()
    
    def run(self):
        """Main enhanced application runner"""
        # Render main interface
        self.render_header()
        self.render_enhanced_sidebar()
        
        # Main content area
        st.markdown("---")
        
        # Show capabilities
        self.render_capabilities()
        st.markdown("---")
        
        # Show quick actions if no messages
        if not st.session_state.messages:
            self.render_enhanced_quick_actions()
        
        # Chat history
        self.render_chat_history()
        
        # Chat input
        self.render_chat_input()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #8E8E93; font-size: 0.9rem;'>"
            "🌟 Enhanced Supernova USLM • Math • Code • Plots • Web Search • "
            f"Powered by AlgoRythm Tech"
            "</div>",
            unsafe_allow_html=True
        )


def main():
    """Main function to run the Enhanced Supernova Web UI"""
    try:
        ui = EnhancedSupernovaWebUI()
        ui.run()
    except Exception as e:
        st.error(f"❌ Enhanced Application Error: {str(e)}")
        st.info("🔄 Please refresh the page to try again.")


if __name__ == "__main__":
    main()
