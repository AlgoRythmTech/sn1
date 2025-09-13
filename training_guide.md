# 🌟 Complete Training Guide: Making Supernova as Capable as Claude

This guide will help you train your Supernova model to have advanced capabilities similar to Claude, including math solving, code execution, and reasoning.

## 📋 Overview

After following this guide, your Supernova model will be able to:
- ✅ Solve mathematical problems (with Wolfram Alpha integration)
- ✅ Write, execute, and debug Python code
- ✅ Create data visualizations and plots  
- ✅ Perform web searches for real-time information
- ✅ Apply safety checks and content filtering
- ✅ Provide company-specific responses (AlgoRythm Tech)
- ✅ Use multi-step reasoning with tools

## 🛠️ Prerequisites

### 1. Install Enhanced Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Additional packages for enhanced capabilities
pip install sympy matplotlib numpy requests streamlit

# Optional: For Wolfram Alpha support
# Get API key from: https://developer.wolframalpha.com/
```

### 2. Set Up API Keys (Optional but Recommended)

```bash
# In your environment or .env file:
export SERPER_API_KEY="your_serper_key"  # For web search
export WOLFRAM_API_KEY="your_wolfram_key"  # For advanced math
```

## 📚 Training Data Requirements

Your model needs training data that teaches it to use tools and reason. Here's the format:

### Enhanced Conversation Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Supernova, an AI assistant with advanced capabilities. You can solve math problems, execute code, create plots, and search the web. When users ask for math, code, or data tasks, use your tools to provide accurate results."
    },
    {
      "role": "user", 
      "content": "Solve the equation x² + 5x + 6 = 0"
    },
    {
      "role": "assistant",
      "content": "I'll solve this quadratic equation for you.\n\nThis is a quadratic equation in the form ax² + bx + c = 0, where:\n- a = 1\n- b = 5  \n- c = 6\n\n🧮 **Mathematical Solution:**\n**Answer:** [-3, -2]\n**Method:** sympy\n**Steps:**\n1. Equation: x² + 5x + 6 = 0\n2. Variable: x\n3. Solutions: [-3, -2]\n\nThe equation factors as (x + 3)(x + 2) = 0, giving us the solutions x = -3 and x = -2."
    }
  ]
}
```

### Key Training Examples You Need

#### 1. Math Problem Solving
```json
{
  "messages": [
    {"role": "user", "content": "What's the derivative of x³ + 2x² - 5?"},
    {"role": "assistant", "content": "I'll find the derivative using calculus rules.\n\n🧮 **Mathematical Solution:**\n**Answer:** 3*x² + 4*x\n**Method:** sympy\n**Steps:**\n1. Expression: x³ + 2x² - 5\n2. Taking derivative with respect to x\n3. Result: 3x² + 4x\n\nUsing the power rule: d/dx[xⁿ] = n·xⁿ⁻¹"}
  ]
}
```

#### 2. Code Generation and Execution
```json
{
  "messages": [
    {"role": "user", "content": "Write Python code to calculate the factorial of 5"},
    {"role": "assistant", "content": "I'll write a Python function to calculate the factorial of 5.\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nresult = factorial(5)\nprint(f\"Factorial of 5 is: {result}\")\n```\n\n💻 **Code Execution Result:**\n```\nFactorial of 5 is: 120\n```\n*Executed in 0.02s*\n\nThe factorial of 5 is 120, calculated as 5 × 4 × 3 × 2 × 1 = 120."}
  ]
}
```

#### 3. Data Analysis and Plotting
```json
{
  "messages": [
    {"role": "user", "content": "Create a plot showing sales data: [10, 25, 30, 45, 20]"},
    {"role": "assistant", "content": "I'll create a visualization of your sales data.\n\n```python\nimport matplotlib.pyplot as plt\n\nsales_data = [10, 25, 30, 45, 20]\ndays = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']\n\nplt.figure(figsize=(10, 6))\nplt.bar(days, sales_data, color='skyblue')\nplt.title('Sales Data Over 5 Days')\nplt.xlabel('Days')\nplt.ylabel('Sales')\nplt.grid(axis='y', alpha=0.3)\nplt.show()\n```\n\n📊 **Plot generated successfully!**\n\nThe chart shows your sales data with Day 4 having the highest sales (45) and Day 1 the lowest (10). There's a general upward trend from Day 1 to Day 4, with a drop on Day 5."}
  ]
}
```

#### 4. Multi-step Reasoning
```json
{
  "messages": [
    {"role": "user", "content": "I have $1000 to invest. If I get 8% annual return, how much will I have after 5 years? Show me the calculation and create a growth chart."},
    {"role": "assistant", "content": "I'll calculate your compound interest and create a visualization.\n\n**Step 1: Calculate Final Amount**\n🧮 **Mathematical Solution:**\nUsing compound interest formula: A = P(1 + r)ⁿ\n- P (Principal) = $1000\n- r (Rate) = 8% = 0.08\n- n (Years) = 5\n\nA = 1000(1.08)⁵ = $1,469.33\n\n**Step 2: Create Growth Visualization**\n\n```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nprincipal = 1000\nrate = 0.08\nyears = np.arange(0, 6)\nvalues = [principal * (1 + rate)**year for year in years]\n\nplt.figure(figsize=(10, 6))\nplt.plot(years, values, marker='o', linewidth=2, color='green')\nplt.title('Investment Growth at 8% Annual Return')\nplt.xlabel('Years')\nplt.ylabel('Investment Value ($)')\nplt.grid(True, alpha=0.3)\nplt.xticks(years)\nfor i, v in enumerate(values):\n    plt.annotate(f'${v:.0f}', (i, v), textcoords=\"offset points\", xytext=(0,10), ha='center')\nplt.show()\n```\n\n📊 **Growth Chart Created!**\n\n**Summary:**\n- Initial Investment: $1,000\n- Final Amount (5 years): $1,469.33\n- Total Gain: $469.33\n- Total Return: 46.93%\n\nYour investment will grow steadily each year with compound interest working in your favor!"}
  ]
}
```

## 🏋️ Training Steps

### Step 1: Prepare Enhanced Training Data

1. **Create comprehensive dataset** with examples showing tool usage
2. **Include error handling** examples
3. **Add safety demonstrations** 
4. **Company knowledge integration**

### Step 2: Run Training with Enhanced System

```bash
# Test the enhanced tools first
python advanced_tools.py

# Run training with enhanced dataset
python training.py \
    --data_path enhanced_training_data.json \
    --model_name supernova-enhanced \
    --batch_size 4 \
    --learning_rate 3e-5 \
    --max_epochs 5 \
    --mask_user_tokens true \
    --mixed_precision true
```

### Step 3: Test Enhanced Model

```bash
# Test with enhanced web UI
python enhanced_web_ui.py

# Or test CLI with tools
python -c "
from advanced_tools import create_enhanced_chat_system
EnhancedChat, tools = create_enhanced_chat_system()
chat = EnhancedChat()
print(chat.chat('Solve x² + 5x + 6 = 0'))
"
```

## 🔧 API Key Setup

### Get Required API Keys

1. **Wolfram Alpha** (for advanced math):
   - Go to https://developer.wolframalpha.com/
   - Sign up and get free API key
   - Add to environment: `WOLFRAM_API_KEY=your_key`

2. **Serper** (for web search):
   - Go to https://serper.dev/
   - Get API key (has free tier)
   - Add to environment: `SERPER_API_KEY=your_key`

### Environment Variables

Create a `.env` file:
```bash
WOLFRAM_API_KEY=your_wolfram_key_here
SERPER_API_KEY=your_serper_key_here
```

## 🚀 After Training - Enhanced Capabilities

### What Your Model Will Be Able To Do:

1. **Advanced Math** 🧮
   - Solve equations (quadratic, cubic, etc.)
   - Calculate derivatives and integrals
   - Handle matrix operations
   - Statistical analysis
   - Use both SymPy and Wolfram Alpha

2. **Code Execution** 💻  
   - Write Python code
   - Execute and test code safely
   - Debug and fix errors
   - Create algorithms
   - Handle data processing

3. **Data Visualization** 📊
   - Create plots and charts
   - Analyze data trends
   - Generate reports
   - Interactive visualizations

4. **Web Intelligence** 🔍
   - Search for current information
   - Fact-checking
   - Research assistance
   - Real-time data

5. **Safety & Company Branding** 🛡️
   - Content filtering
   - AlgoRythm Tech responses
   - Professional interactions
   - Ethical guidelines

## 🎯 Usage Examples After Training

### Math Problems
```
User: "Find the area under the curve y = x² from 0 to 3"
Supernova: [Calculates integral using Wolfram Alpha/SymPy]
```

### Coding Tasks  
```
User: "Write a sorting algorithm and test it with [3,1,4,1,5]"
Supernova: [Writes code, executes it, shows results]
```

### Data Analysis
```
User: "Analyze this sales data and show trends"
Supernova: [Processes data, creates visualizations, provides insights]
```

### Research Questions
```
User: "What's the current price of Bitcoin?"
Supernova: [Searches web, provides current information]
```

## 🔄 Iterative Improvement

### Training Loop:
1. **Train** with enhanced dataset
2. **Test** capabilities in web UI  
3. **Identify** gaps or errors
4. **Add** more training examples for weak areas
5. **Retrain** and improve
6. **Repeat** until satisfied

### Monitoring:
- Use the web UI to test different scenarios
- Check tool integration works correctly
- Verify safety measures are active
- Ensure company responses are appropriate

## 🎉 Final Result

After completing this training guide, your Supernova model will have capabilities very similar to Claude:

✅ **Mathematical Problem Solving** (Wolfram Alpha level)  
✅ **Code Generation & Execution** (Python with safety)  
✅ **Data Analysis & Visualization** (Matplotlib integration)  
✅ **Web Search & Research** (Real-time information)  
✅ **Multi-step Reasoning** (Chain of thought with tools)  
✅ **Safety & Content Filtering** (Responsible AI)  
✅ **Company Knowledge** (AlgoRythm Tech branding)  

The enhanced web UI provides a beautiful interface to interact with all these capabilities, making your Supernova model a powerful AI assistant ready for professional use!

---

**Need Help?** Check the troubleshooting section in the README or test individual components using the provided scripts.
