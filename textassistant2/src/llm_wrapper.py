from langchain_nvidia_ai_endpoints import ChatNVIDIA

client = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1-0528",
    api_key="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",
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
)

def generate(prompt):
    response = ""
    for chunk in client.stream([{"role": "user", "content": prompt}]):
        if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
            response += chunk.additional_kwargs["reasoning_content"]
        response += chunk.content
    return response
