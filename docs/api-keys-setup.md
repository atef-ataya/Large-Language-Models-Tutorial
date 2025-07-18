# 🔑 API Keys Setup Guide

This guide walks you through setting up API keys for all the tutorials. Most LLM applications require external services, and this guide makes the process simple and secure.

## 🚀 Quick Setup Checklist

- [ ] OpenAI API Key (required for most tutorials)
- [ ] Pinecone API Key (for vector database tutorials)
- [ ] Tavily API Key (for research agent tutorials)
- [ ] Hugging Face Token (for some models)
- [ ] Environment setup (local or cloud)

## 🔑 Required API Keys

### 1. OpenAI API Key (Essential)

**Used in:** Almost all tutorials (ChatGPT, GPT-4, embeddings)

#### Getting Your Key

1. **Visit** [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Sign up** or login to your account
3. **Navigate** to "API Keys" section
4. **Click** "Create new secret key"
5. **Name** your key (e.g., "LLM-Tutorials")
6. **Copy** the key immediately (you won't see it again!)

#### Pricing Info

- **Free Tier**: $5 credit for new accounts
- **Pay-as-you-go**: ~$0.002 per 1K tokens for GPT-3.5
- **Usage Tracking**: Monitor at [platform.openai.com/usage](https://platform.openai.com/usage)

```python
# Example usage cost for tutorials
# Basic tutorial: ~$0.01 - $0.05
# Complex agent tutorial: ~$0.10 - $0.50
```

### 2. Pinecone API Key (Vector Databases)

**Used in:** RAG systems, semantic search, chatbots with memory

#### Getting Your Key

1. **Visit** [Pinecone Console](https://app.pinecone.io/)
2. **Create** a free account
3. **Go to** "API Keys" in the sidebar
4. **Copy** your API key
5. **Note** your environment (e.g., "us-east1-gcp")

#### Free Tier

- **Indexes**: 1 index
- **Dimensions**: Up to 1536 dimensions
- **Vectors**: 100K vectors
- **Perfect** for learning and prototyping!

### 3. Tavily API Key (AI Research)

**Used in:** Research agents, web search integration

#### Getting Your Key

1. **Visit** [Tavily](https://tavily.com/)
2. **Sign up** for an account
3. **Navigate** to dashboard
4. **Generate** API key
5. **Copy** the key

#### Features

- **Web Search**: Real-time web search
- **Content Extraction**: Clean, structured data
- **Free Tier**: 1000 requests/month

### 4. Hugging Face Token (Optional)

**Used in:** Open-source models, model downloads

#### Getting Your Token

1. **Visit** [Hugging Face](https://huggingface.co/settings/tokens)
2. **Login** to your account
3. **Create** new token
4. **Select** "Read" permissions
5. **Copy** the token

## 🛠️ Setting Up API Keys

### 🌟 Method 1: Google Colab (Recommended for Beginners)

**Secure and Easy Setup:**

```python
# In Google Colab, use the built-in secrets management
from google.colab import userdata
import os

# Add your keys in Colab's Secrets panel (🔑 icon in sidebar)
# Then access them securely:
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ["PINECONE_API_KEY"] = userdata.get('PINECONE_API_KEY')
os.environ["PINECONE_ENVIRONMENT"] = userdata.get('PINECONE_ENVIRONMENT')
os.environ["TAVILY_API_KEY"] = userdata.get('TAVILY_API_KEY')

print("✅ API keys loaded successfully!")
```

**How to add secrets in Colab:**

1. Click the 🔑 (key) icon in the left sidebar
2. Click "Add new secret"
3. Name: `OPENAI_API_KEY`, Value: `your-actual-key`
4. Repeat for other keys

### 🖥️ Method 2: Local Development

#### Option A: Environment Variables (Recommended)

**Create a `.env` file:**

```bash
# .env file (never commit this to git!)
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_ENVIRONMENT=us-east1-gcp
TAVILY_API_KEY=your-tavily-key-here
HUGGINGFACE_TOKEN=your-hf-token-here
```

**Load in Python:**

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify keys are loaded
required_keys = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
for key in required_keys:
    if os.getenv(key):
        print(f"✅ {key} loaded")
    else:
        print(f"❌ {key} missing")
```

**Install python-dotenv:**

```bash
pip install python-dotenv
```

#### Option B: System Environment Variables

**On macOS/Linux:**

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="your-openai-key"
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_ENVIRONMENT="us-east1-gcp"

# Reload shell
source ~/.bashrc
```

**On Windows:**

```cmd
# Command Prompt
setx OPENAI_API_KEY "your-openai-key"
setx PINECONE_API_KEY "your-pinecone-key"

# PowerShell
$env:OPENAI_API_KEY="your-openai-key"
$env:PINECONE_API_KEY="your-pinecone-key"
```

#### Option C: Direct Assignment (Not Recommended)

```python
# Only for testing - never commit keys to code!
import os

os.environ["OPENAI_API_KEY"] = "sk-your-actual-key"  # Replace with real key
os.environ["PINECONE_API_KEY"] = "your-actual-key"   # Replace with real key
```

### ☁️ Method 3: GitHub Codespaces

1. **Open** your repository in Codespaces
2. **Go to** Settings → Codespaces
3. **Add** repository secrets:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_ENVIRONMENT`
   - `TAVILY_API_KEY`

## 🔒 Security Best Practices

### ✅ Do's

- **Use environment variables** or secure secret managers
- **Rotate keys regularly** (every 3-6 months)
- **Monitor usage** to detect unauthorized access
- **Use minimal permissions** (read-only when possible)
- **Keep backups** of working configurations

### ❌ Don'ts

- **Never commit keys** to version control
- **Don't share keys** in screenshots or messages
- **Avoid hardcoding** keys in source code
- **Don't use production keys** for testing
- **Never post keys** in forums or chat

### 🛡️ Additional Security

**Add to `.gitignore`:**

```bash
# .gitignore
.env
.env.local
.env.*.local
**/api_keys.py
secrets.json
```

**Key rotation reminder:**

```python
# Add to your notebooks
from datetime import datetime, timedelta

def check_key_age(key_created_date):
    if datetime.now() - key_created_date > timedelta(days=90):
        print("⚠️  Consider rotating your API keys for better security")
```

## 🧪 Testing Your Setup

### Quick Test Script

```python
# test_api_setup.py
import os
from datetime import datetime

def test_api_keys():
    """Test if all required API keys are properly configured"""

    print("🔍 Testing API Key Configuration\n")

    # Required keys
    keys_to_test = {
        "OPENAI_API_KEY": "OpenAI API",
        "PINECONE_API_KEY": "Pinecone Vector DB",
        "PINECONE_ENVIRONMENT": "Pinecone Environment",
    }

    # Optional keys
    optional_keys = {
        "TAVILY_API_KEY": "Tavily Research API",
        "HUGGINGFACE_TOKEN": "Hugging Face Token",
    }

    # Test required keys
    all_good = True
    for key, service in keys_to_test.items():
        value = os.getenv(key)
        if value:
            # Mask the key for security
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {service}: {masked}")
        else:
            print(f"❌ {service}: Not found")
            all_good = False

    # Test optional keys
    print("\n📋 Optional Keys:")
    for key, service in optional_keys.items():
        value = os.getenv(key)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {service}: {masked}")
        else:
            print(f"⚪ {service}: Not configured")

    # Test actual API connectivity
    print("\n🌐 Testing API Connectivity:")

    # Test OpenAI
    try:
        from openai import OpenAI
        client = OpenAI()
        models = client.models.list()
        print("✅ OpenAI API: Connected successfully")
    except Exception as e:
        print(f"❌ OpenAI API: Connection failed - {str(e)[:50]}...")
        all_good = False

    # Test Pinecone
    try:
        import pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        print("✅ Pinecone: Connected successfully")
    except Exception as e:
        print(f"⚪ Pinecone: {str(e)[:50]}...")

    print(f"\n{'🎉 All systems ready!' if all_good else '⚠️  Some keys missing - check above'}")
    return all_good

# Run the test
if __name__ == "__main__":
    test_api_keys()
```

### Quick OpenAI Test

```python
# Quick test to verify OpenAI setup
import os
from openai import OpenAI

def quick_openai_test():
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello from the tutorial setup!'"}],
            max_tokens=10
        )
        print("✅ OpenAI Test:", response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"❌ OpenAI Test Failed: {e}")
        return False

quick_openai_test()
```

## 💰 Cost Management

### Monitor Your Usage

```python
# Check OpenAI usage programmatically
import requests
import os

def check_openai_usage():
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    # Note: This endpoint may require special permissions
    try:
        response = requests.get(
            "https://api.openai.com/v1/usage",
            headers=headers
        )
        if response.status_code == 200:
            print("💰 Usage data:", response.json())
        else:
            print("📊 Check usage at: https://platform.openai.com/usage")
    except:
        print("📊 Monitor usage at: https://platform.openai.com/usage")

check_openai_usage()
```

### Set Usage Alerts

1. **OpenAI**: Set up usage alerts in your dashboard
2. **Pinecone**: Monitor index usage
3. **Budget Tracking**: Keep a spreadsheet of API costs

## 🔄 Key Rotation Schedule

```python
# Set reminders for key rotation
import datetime

def key_rotation_reminder():
    # Set up key rotation reminders
    creation_dates = {
        "openai": datetime.date(2024, 1, 1),  # Update with your actual dates
        "pinecone": datetime.date(2024, 1, 1),
        "tavily": datetime.date(2024, 1, 1)
    }

    today = datetime.date.today()

    for service, created in creation_dates.items():
        days_old = (today - created).days
        if days_old > 90:
            print(f"⚠️  {service.title()} key is {days_old} days old - consider rotating")
        else:
            print(f"✅ {service.title()} key is fresh ({days_old} days old)")

key_rotation_reminder()
```

## 📚 Tutorial-Specific Setup

### Basic Tutorials (LangChain 101, Prompt Engineering)

**Required:**

- OpenAI API Key

### RAG & Chatbot Tutorials

**Required:**

- OpenAI API Key
- Pinecone API Key + Environment

### Research Agent Tutorials

**Required:**

- OpenAI API Key
- Tavily API Key

### Advanced Multi-Agent Tutorials

**Required:**

- OpenAI API Key
- Pinecone API Key
- Tavily API Key

### Speech & Image Tutorials

**Required:**

- OpenAI API Key (for Whisper and DALL-E)

## 🚨 Troubleshooting API Issues

### Common Error Messages

**"Invalid API key"**

```python
# Debug steps
import os
key = os.getenv("OPENAI_API_KEY")
print(f"Key exists: {key is not None}")
print(f"Key starts with 'sk-': {key.startswith('sk-') if key else False}")
print(f"Key length: {len(key) if key else 0} characters")
```

**"Rate limit exceeded"**

```python
# Add delays between requests
import time
time.sleep(1)  # Wait 1 second between API calls

# Or implement exponential backoff
import time
import random

def api_call_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                raise e
```

**"Insufficient quota"**

- Check your [OpenAI billing](https://platform.openai.com/account/billing)
- Add payment method if using free credits
- Monitor usage to avoid surprises

**"Model not found"**

```python
# List available models
from openai import OpenAI
client = OpenAI()
models = client.models.list()
for model in models.data:
    print(model.id)
```

### Environment-Specific Troubleshooting

**Google Colab:**

```python
# If secrets aren't working, try manual setup
import os
import getpass

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API Key: ")
```

**Local Jupyter:**

```python
# Restart kernel after setting environment variables
import importlib
import sys

# Clear module cache
for module in list(sys.modules.keys()):
    if module.startswith('openai') or module.startswith('langchain'):
        del sys.modules[module]

# Reimport
from openai import OpenAI
```

## 📖 Additional Resources

### Official Documentation

- **OpenAI API**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **Pinecone**: [docs.pinecone.io](https://docs.pinecone.io)
- **LangChain**: [python.langchain.com](https://python.langchain.com)
- **Tavily**: [docs.tavily.com](https://docs.tavily.com)

### Community Resources

- **OpenAI Community**: [community.openai.com](https://community.openai.com)
- **LangChain Discord**: [discord.gg/langchain](https://discord.gg/langchain)
- **r/MachineLearning**: [reddit.com/r/MachineLearning](https://reddit.com/r/MachineLearning)

### Security Resources

- **API Security Best Practices**: [owasp.org/www-project-api-security/](https://owasp.org/www-project-api-security/)
- **Environment Variables Guide**: [12factor.net/config](https://12factor.net/config)

## 🎯 Next Steps

Once your API keys are set up:

1. **Test** with a simple tutorial like "LangChain 101"
2. **Monitor** your usage for the first few days
3. **Set up** usage alerts and budgets
4. **Bookmark** the troubleshooting section
5. **Join** the community for ongoing support

## 📞 Getting Help

If you're still having trouble:

1. **Check** the [troubleshooting guide](troubleshooting.md)
2. **Search** existing [GitHub issues](https://github.com/atef-ataya/Large-Language-Models-Tutorial/issues)
3. **Create** a new issue with:
   - Which API key is causing trouble
   - Error messages (remove actual keys!)
   - Environment details (Colab, local, etc.)
4. **Ask** on [Twitter](https://twitter.com/atef_ataya)

---

**🔐 Remember: Your API keys are like passwords - keep them safe and never share them!**

_Ready to start building? Head to the [Quick Start Guide](quick-start.md)!_ 🚀
