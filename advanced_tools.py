"""
Advanced Tools System for Supernova USLM
Provides enhanced capabilities like math solving, code execution, and more
"""

import os
import re
import json
import subprocess
import tempfile
import requests
from typing import Dict, List, Optional, Any, Tuple
import sympy
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import ast
import sys
from pathlib import Path

class AdvancedToolSystem:
    """Enhanced capabilities system for Supernova"""
    
    def __init__(self, wolfram_api_key: Optional[str] = None):
        self.wolfram_api_key = wolfram_api_key or os.getenv('WOLFRAM_API_KEY')
        self.temp_dir = tempfile.mkdtemp(prefix='supernova_')
        
        # Tool availability
        self.tools = {
            'math': self._has_math_tools(),
            'code': self._has_code_tools(),
            'wolfram': bool(self.wolfram_api_key),
            'plotting': self._has_plotting_tools(),
            'web': True  # Already implemented in web_search.py
        }
        
        print(f"🛠️  Advanced Tools Initialized:")
        for tool, available in self.tools.items():
            status = "✅" if available else "❌"
            print(f"   {status} {tool.upper()}")
    
    def _has_math_tools(self) -> bool:
        """Check if math libraries are available"""
        try:
            import sympy
            import numpy
            return True
        except ImportError:
            return False
    
    def _has_code_tools(self) -> bool:
        """Check if code execution tools are available"""
        return True  # Python is always available
    
    def _has_plotting_tools(self) -> bool:
        """Check if plotting libraries are available"""
        try:
            import matplotlib
            return True
        except ImportError:
            return False
    
    def solve_math(self, problem: str, use_wolfram: bool = True) -> Dict[str, Any]:
        """
        Solve mathematical problems using SymPy or Wolfram Alpha
        """
        result = {
            'success': False,
            'solution': None,
            'steps': [],
            'method': None,
            'error': None
        }
        
        try:
            # Try Wolfram Alpha first if available and requested
            if use_wolfram and self.tools['wolfram']:
                wolfram_result = self._query_wolfram(problem)
                if wolfram_result['success']:
                    result.update(wolfram_result)
                    result['method'] = 'wolfram_alpha'
                    return result
            
            # Fall back to SymPy
            if self.tools['math']:
                sympy_result = self._solve_with_sympy(problem)
                if sympy_result['success']:
                    result.update(sympy_result)
                    result['method'] = 'sympy'
                    return result
        
        except Exception as e:
            result['error'] = f"Math solving error: {str(e)}"
        
        return result
    
    def _query_wolfram(self, query: str) -> Dict[str, Any]:
        """Query Wolfram Alpha API"""
        if not self.wolfram_api_key:
            return {'success': False, 'error': 'Wolfram API key not available'}
        
        try:
            url = "http://api.wolframalpha.com/v2/query"
            params = {
                'input': query,
                'format': 'plaintext',
                'output': 'JSON',
                'appid': self.wolfram_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if data.get('queryresult', {}).get('success'):
                    pods = data['queryresult'].get('pods', [])
                    solution = None
                    steps = []
                    
                    for pod in pods:
                        if pod.get('primary'):
                            subpods = pod.get('subpods', [])
                            if subpods:
                                solution = subpods[0].get('plaintext', '')
                        
                        # Collect step-by-step solutions
                        if 'step' in pod.get('title', '').lower():
                            subpods = pod.get('subpods', [])
                            for subpod in subpods:
                                step_text = subpod.get('plaintext', '')
                                if step_text:
                                    steps.append(step_text)
                    
                    return {
                        'success': True,
                        'solution': solution,
                        'steps': steps
                    }
            
            return {'success': False, 'error': 'Wolfram API request failed'}
            
        except Exception as e:
            return {'success': False, 'error': f"Wolfram API error: {str(e)}"}
    
    def _solve_with_sympy(self, problem: str) -> Dict[str, Any]:
        """Solve math problems using SymPy"""
        try:
            # Clean up the input
            problem = problem.replace('solve', '').replace('find', '').strip()
            
            # Try to parse as equation
            if '=' in problem:
                left, right = problem.split('=', 1)
                equation = f"{left.strip()} - ({right.strip()})"
            else:
                equation = problem
            
            # Parse with SymPy
            expr = sympy.sympify(equation)
            
            # Get free symbols (variables)
            variables = list(expr.free_symbols)
            
            if variables:
                # Solve equation
                solutions = sympy.solve(expr, variables[0])
                
                return {
                    'success': True,
                    'solution': str(solutions),
                    'steps': [f"Equation: {expr} = 0", f"Variable: {variables[0]}", f"Solutions: {solutions}"]
                }
            else:
                # Evaluate expression
                result = sympy.simplify(expr)
                return {
                    'success': True,
                    'solution': str(result),
                    'steps': [f"Expression: {expr}", f"Simplified: {result}"]
                }
                
        except Exception as e:
            return {'success': False, 'error': f"SymPy error: {str(e)}"}
    
    def execute_code(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """
        Execute code safely and return results
        """
        result = {
            'success': False,
            'output': None,
            'error': None,
            'execution_time': None
        }
        
        if language.lower() != 'python':
            result['error'] = f"Language '{language}' not supported yet. Only Python is available."
            return result
        
        try:
            # Security check - basic unsafe patterns
            unsafe_patterns = [
                'import os', 'import sys', 'import subprocess', 'import shutil',
                '__import__', 'exec', 'eval', 'open(', 'file(',
                'input(', 'raw_input('
            ]
            
            code_lower = code.lower()
            for pattern in unsafe_patterns:
                if pattern in code_lower:
                    result['error'] = f"Security: Code contains potentially unsafe operation: {pattern}"
                    return result
            
            # Execute in temporary file
            start_time = datetime.now()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self.temp_dir) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Capture output
                process = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10,  # 10 second timeout
                    cwd=self.temp_dir
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                if process.returncode == 0:
                    result.update({
                        'success': True,
                        'output': process.stdout,
                        'execution_time': execution_time
                    })
                else:
                    result.update({
                        'success': False,
                        'error': process.stderr,
                        'execution_time': execution_time
                    })
                
            finally:
                # Clean up
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            result['error'] = "Code execution timed out (10s limit)"
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
        
        return result
    
    def create_plot(self, data_or_code: str, plot_type: str = 'auto') -> Dict[str, Any]:
        """
        Create plots from data or plotting code
        """
        if not self.tools['plotting']:
            return {
                'success': False,
                'error': 'Matplotlib not available for plotting'
            }
        
        result = {
            'success': False,
            'image_data': None,
            'error': None
        }
        
        try:
            # Create a figure
            plt.figure(figsize=(10, 6))
            
            # Execute the plotting code
            if 'plt.' in data_or_code or 'matplotlib' in data_or_code:
                # Direct plotting code
                exec(data_or_code)
            else:
                # Try to parse as data and create a simple plot
                try:
                    # Simple data parsing
                    data = eval(data_or_code)
                    if isinstance(data, (list, tuple)):
                        plt.plot(data)
                        plt.title("Data Plot")
                        plt.grid(True)
                except:
                    result['error'] = "Could not parse data for plotting"
                    return result
            
            # Save plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            result.update({
                'success': True,
                'image_data': image_data
            })
            
        except Exception as e:
            result['error'] = f"Plotting error: {str(e)}"
            plt.close()  # Ensure figure is closed
        
        return result
    
    def analyze_text(self, text: str, analysis_type: str = 'summary') -> Dict[str, Any]:
        """
        Perform text analysis
        """
        result = {
            'success': False,
            'analysis': None,
            'error': None
        }
        
        try:
            if analysis_type == 'summary':
                # Simple extractive summary
                sentences = text.split('.')
                word_count = len(text.split())
                char_count = len(text)
                
                result.update({
                    'success': True,
                    'analysis': {
                        'word_count': word_count,
                        'character_count': char_count,
                        'sentence_count': len(sentences),
                        'avg_words_per_sentence': word_count / max(len(sentences), 1),
                        'first_sentence': sentences[0].strip() if sentences else "",
                        'last_sentence': sentences[-1].strip() if sentences and sentences[-1].strip() else ""
                    }
                })
            
            elif analysis_type == 'keywords':
                # Simple keyword extraction
                words = text.lower().split()
                word_freq = {}
                for word in words:
                    word = re.sub(r'[^\w]', '', word)
                    if len(word) > 3:  # Ignore short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Top 10 keywords
                keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                
                result.update({
                    'success': True,
                    'analysis': {
                        'keywords': keywords,
                        'unique_words': len(word_freq),
                        'total_words': len(words)
                    }
                })
            
        except Exception as e:
            result['error'] = f"Text analysis error: {str(e)}"
        
        return result
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return current capabilities and status
        """
        return {
            'tools_available': self.tools,
            'math_solving': {
                'sympy': self.tools['math'],
                'wolfram_alpha': self.tools['wolfram'],
                'supported_operations': [
                    'algebraic equations', 'calculus', 'linear algebra',
                    'trigonometry', 'statistics', 'number theory'
                ]
            },
            'code_execution': {
                'python': True,
                'safety_checks': True,
                'timeout': '10 seconds',
                'supported_libraries': ['numpy', 'matplotlib', 'sympy']
            },
            'plotting': {
                'available': self.tools['plotting'],
                'formats': ['PNG (base64)'],
                'types': ['line plots', 'bar charts', 'scatter plots', 'histograms']
            },
            'text_analysis': {
                'available': True,
                'types': ['summary statistics', 'keyword extraction']
            }
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass


# Integration functions for the chat interface
def enhance_response_with_tools(user_input: str, base_response: str, tools: AdvancedToolSystem) -> str:
    """
    Enhance a response using available tools when appropriate
    """
    enhanced_response = base_response
    
    # Check if user is asking for math
    math_patterns = [
        r'solve\s+.*equation',
        r'calculate\s+.*',
        r'what\s+is\s+\d+.*[\+\-\*\/].*\d+',
        r'find\s+.*=',
        r'\d+\s*[\+\-\*\/]\s*\d+',
        r'integral|derivative|limit',
        r'matrix|determinant'
    ]
    
    user_lower = user_input.lower()
    
    # Math detection
    for pattern in math_patterns:
        if re.search(pattern, user_lower):
            math_result = tools.solve_math(user_input)
            if math_result['success']:
                enhanced_response += f"\n\n🧮 **Mathematical Solution:**\n"
                enhanced_response += f"**Answer:** {math_result['solution']}\n"
                if math_result['steps']:
                    enhanced_response += f"**Method:** {math_result['method']}\n"
                    enhanced_response += f"**Steps:**\n"
                    for i, step in enumerate(math_result['steps'], 1):
                        enhanced_response += f"{i}. {step}\n"
            break
    
    # Code detection
    if 'write code' in user_lower or 'python code' in user_lower or '```python' in user_input:
        # Extract code if present
        code_match = re.search(r'```python\n(.*?)\n```', user_input, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            exec_result = tools.execute_code(code)
            if exec_result['success']:
                enhanced_response += f"\n\n💻 **Code Execution Result:**\n"
                enhanced_response += f"```\n{exec_result['output']}\n```\n"
                enhanced_response += f"*Executed in {exec_result['execution_time']:.2f}s*"
            elif exec_result['error']:
                enhanced_response += f"\n\n❌ **Code Execution Error:**\n"
                enhanced_response += f"```\n{exec_result['error']}\n```"
    
    # Plot detection
    if 'plot' in user_lower or 'graph' in user_lower or 'chart' in user_lower:
        if tools.tools['plotting']:
            enhanced_response += f"\n\n📊 **Plotting capability available!** Use the web interface to generate plots."
    
    return enhanced_response


def create_enhanced_chat_system() -> Tuple[Any, AdvancedToolSystem]:
    """
    Create an enhanced chat system with advanced tools
    """
    from chat_interface import SupernovaChat
    
    # Initialize tools
    tools = AdvancedToolSystem()
    
    # Create enhanced chat class
    class EnhancedSupernovaChat(SupernovaChat):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tools = tools
        
        def chat(self, user_input: str, enhance_with_tools: bool = True, **kwargs) -> str:
            # Get base response
            base_response = super().chat(user_input, **kwargs)
            
            # Enhance with tools if requested
            if enhance_with_tools:
                return enhance_response_with_tools(user_input, base_response, self.tools)
            
            return base_response
        
        def get_capabilities(self) -> Dict[str, Any]:
            """Get enhanced capabilities"""
            base_caps = super().get_model_info()
            tool_caps = self.tools.get_capabilities()
            
            return {
                'model': base_caps,
                'advanced_tools': tool_caps
            }
    
    return EnhancedSupernovaChat, tools


if __name__ == "__main__":
    # Test the tools system
    print("🧪 Testing Advanced Tools System...\n")
    
    tools = AdvancedToolSystem()
    
    # Test math
    print("Testing math solving...")
    result = tools.solve_math("solve x^2 + 5x + 6 = 0")
    print(f"Math result: {result}\n")
    
    # Test code execution
    print("Testing code execution...")
    code = """
import math
print("Hello from Supernova!")
result = math.sqrt(16)
print(f"Square root of 16 is: {result}")
"""
    result = tools.execute_code(code)
    print(f"Code result: {result}\n")
    
    # Test capabilities
    print("Available capabilities:")
    caps = tools.get_capabilities()
    print(json.dumps(caps, indent=2))
    
    # Cleanup
    tools.cleanup()
