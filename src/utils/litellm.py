import os
from litellm import completion

MODEL = "gemini/gemini-2.5-flash-lite"
PROXY_URL = "http://localhost:4000/v1"
PROXY_KEY = os.getenv("LITELLM_MASTER_KEY")


def run_model(content: list) -> str | None:
    try:
        respone = completion(
            model=MODEL,
            proxy_url=PROXY_URL,
            proxy_key=PROXY_KEY,
            messages=[{"role": "user", "content": content}],
        )

    except Exception as e:
        print(f"Error running model: {e}")
        raise e

    print(respone.choices[0].message.content)
    return respone.choices[0].message.content
