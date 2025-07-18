# 🔧 Troubleshooting Guide

Encountering issues? This guide covers the most common problems and their solutions.

## 🚨 Common Issues & Quick Fixes

### 🔴 Installation & Setup Issues

#### Problem: "ModuleNotFoundError" when importing libraries

```python
ModuleNotFoundError: No module named 'langchain'
```

**Solutions:**

```bash
# Option 1: Install missing package
pip install langchain

# Option 2: Install all requirements
pip install -r requirements.txt

# Option 3: Force reinstall
pip install --force-reinstall langchain

# Option 4: In Google Colab
!pip install langchain
```

#### Problem: Python version compatibility

```
ERROR: Package requires Python '>=3.8' but you have Python '3.7'
```

**Solutions:**

- **Local**: Update Python to 3.8+ from [python.org](https://python.org)
- **Colab**: Use Google Colab (always up-to-date)
- **Conda**: `conda install python=3.8`

#### Problem: Jupyter Notebook won't start

```bash
# Try these commands
jupyter notebook --generate-config
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

### 🔐 API Key Issues

#### Problem: "Invalid API key" or "Authentication failed"

```python
AuthenticationError: Invalid API key provided
```

**Solutions:**

1. **Check API Key Format:**

   ```python
   # OpenAI keys start with 'sk-'
   # Pinecone keys are usually 32 characters
   print(f"OpenAI key starts with 'sk-': {os.environ['OPENAI_API_KEY'].startswith('sk-')}")
   ```

2. **Verify Key is Active:**

   - Visit [OpenAI Dashboard](https://platform.openai.com/account/api-keys)
   - Check if key is still valid and has credits

3. **Environment Variable Setup:**

   ```python
   import os

   # Method 1: Direct assignment
   os.environ["OPENAI_API_KEY"] = "your-actual-key-here"

   # Method 2: Using python-dotenv
   from dotenv import load_dotenv
   load_dotenv()  # Loads from .env file
   ```

#### Problem: "Rate limit exceeded"

```python
RateLimitError: Rate limit reached for requests
```

**Solutions:**

- **Wait**: Most limits reset after a few minutes
- **Upgrade Plan**: Consider OpenAI paid plan for higher limits
- **Add Delays**: Use `time.sleep(1)` between API calls
- **Batch Requests**: Process data in smaller chunks

### 🖥️ Environment-Specific Issues

#### Google Colab Issues

**Problem: Runtime disconnected**

- **Cause**: Colab has usage limits and timeouts
- **Solution**:
  ```python
  # Keep session alive
  import time
  while True:
      time.sleep(60)  # Check every minute
  ```

**Problem: Files disappear after session**

- **Cause**: Colab resets after inactivity
- **Solution**: Save important files to Google Drive

  ```python
  from google.colab import drive
  drive.mount('/content/drive')

  # Save your work
  with open('/content/drive/MyDrive/my_results.txt', 'w') as f:
      f.write("Your results here")
  ```

#### Local Jupyter Issues

**Problem: Kernel keeps dying**

```bash
# Check memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Reduce memory usage
del large_variable  # Delete large objects
import gc; gc.collect()  # Force garbage collection
```

**Problem: Can't access localhost:8888**

```bash
# Try different port
jupyter notebook --port=8889

# Check what's using the port
netstat -an | grep 8888
```

### 🤖 LLM-Specific Issues

#### Problem: Poor response quality

```python
# Your prompt might be too vague
bad_prompt = "Explain AI"
good_prompt = """
Explain artificial intelligence in simple terms for a beginner,
including 3 practical examples and how it differs from traditional programming.
Keep the response under 200 words.
"""
```

**Solutions:**

- **Be Specific**: Add context, format requirements, and examples
- **Use System Messages**: Set behavior expectations
- **Temperature Control**: Lower for consistent responses, higher for creativity
- **Try Different Models**: GPT-4 vs GPT-3.5 vs others

#### Problem: Inconsistent results

```python
# Add temperature control
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Your prompt"}],
    temperature=0.1,  # Lower = more consistent
    seed=42  # Some models support seeding
)
```

#### Problem: Context length exceeded

```python
# Token counting and management
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Truncate if needed
max_tokens = 4000  # Leave room for response
if count_tokens(your_text) > max_tokens:
    # Truncate or summarize
    your_text = your_text[:max_tokens * 4]  # Rough character limit
```

### 🔗 LangChain-Specific Issues

#### Problem: "No default OpenAI API key found"

```python
# Ensure environment variables are set BEFORE importing LangChain
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# THEN import LangChain
from langchain.llms import OpenAI
```

#### Problem: Vector store connection issues

```python
# Pinecone connection troubleshooting
import pinecone

# Initialize with proper error handling
try:
    pinecone.init(
        api_key="your-api-key",
        environment="your-env"  # e.g., "us-east1-gcp"
    )
    print("✅ Pinecone connected successfully")
except Exception as e:
    print(f"❌ Pinecone connection failed: {e}")
```

#### Problem: Memory errors with large documents

```python
# Process documents in chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_text(large_document)
```

## 🛠️ Advanced Debugging

### Memory Issues

```python
# Monitor memory usage
import tracemalloc
tracemalloc.start()

# Your code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

### Network Issues

```python
# Test API connectivity
import requests

def test_openai_connection():
    try:
        response = requests.get("https://api.openai.com/v1/models",
                              headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        if response.status_code == 200:
            print("✅ OpenAI API accessible")
        else:
            print(f"❌ OpenAI API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Network error: {e}")

test_openai_connection()
```

### Logging & Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# LangChain specific debugging
import langchain
langchain.debug = True
```

## 📊 Performance Optimization

### Speed Up Notebook Execution

```python
# Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_operation(input_text):
    # Your expensive computation
    return result

# Use async for API calls
import asyncio
import aiohttp

async def async_api_call(prompt):
    # Async API calls for better performance
    pass
```

### Reduce API Costs

```python
# Estimate costs before running
def estimate_cost(text, model="gpt-3.5-turbo"):
    tokens = count_tokens(text)
    cost_per_1k = 0.002  # GPT-3.5-turbo pricing
    estimated_cost = (tokens / 1000) * cost_per_1k
    return estimated_cost

print(f"Estimated cost: ${estimate_cost(your_prompt):.4f}")
```

## 🆘 When All Else Fails

### Create a Minimal Reproducible Example

```python
# Simplify your code to the bare minimum
import os
os.environ["OPENAI_API_KEY"] = "your-key"

from langchain.llms import OpenAI
llm = OpenAI()
response = llm("Hello, world!")
print(response)
```

### Environment Reset

```bash
# Nuclear option: Fresh start
pip freeze > old_requirements.txt
pip uninstall -r old_requirements.txt -y
pip install -r requirements.txt
```

### Get Help from the Community

1. **Search Existing Issues**: [GitHub Issues](https://github.com/atef-ataya/Large-Language-Models-Tutorial/issues)
2. **Create New Issue**: Use our issue templates
3. **Join Discussions**: [GitHub Discussions](https://github.com/atef-ataya/Large-Language-Models-Tutorial/discussions)
4. **Twitter**: [@atef_ataya](https://twitter.com/atef_ataya)

### Include in Your Help Request

- Python version: `python --version`
- Operating system
- Exact error message (copy-paste)
- Code that reproduces the issue
- What you expected vs what happened

## 📚 Additional Resources

- **LangChain Documentation**: [docs.langchain.com](https://docs.langchain.com)
- **OpenAI API Reference**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **Jupyter Troubleshooting**: [jupyter.readthedocs.io](https://jupyter.readthedocs.io)

---

**Still stuck?** Don't hesitate to open an issue! The community is here to help. 🤝

_Remember: Every expert was once a beginner who never gave up!_ 💪
