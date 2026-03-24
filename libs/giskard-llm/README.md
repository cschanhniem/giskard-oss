# giskard-llm

Lightweight LLM routing layer over native provider SDKs. Routes `provider/model` strings to OpenAI, Google Gemini, or Anthropic using their native async SDKs.

## Installation

```bash
pip install giskard-llm[openai]      # OpenAI only
pip install giskard-llm[google]      # Google Gemini only
pip install giskard-llm[anthropic]   # Anthropic only
pip install giskard-llm[all]         # All providers
```

## Usage

```python
from giskard.llm import acompletion, aembedding

response = await acompletion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

embeddings = await aembedding(
    model="openai/text-embedding-3-small",
    input=["hello world"],
)
```
