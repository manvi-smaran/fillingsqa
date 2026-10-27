import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAIError, OpenAI

load_dotenv()

PROMPT = "In one sentence, what is a 10-K filing?"
MAX_TOKENS = 300


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    model: str


@dataclass(frozen=True)
class Result:
    provider: str
    model: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    text: str


PROVIDERS = (
    Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        model="gemini-3.5-flash-lite",
    ),
    Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="openai/gpt-oss-20b",
    ),
)


def ask(provider: Provider, api_key: str, prompt: str) -> Result:
    client = OpenAI(api_key=api_key, base_url=provider.base_url)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=provider.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    latency_s = time.perf_counter() - start

    usage = response.usage
    return Result(
        provider=provider.name,
        model=provider.model,
        latency_s=latency_s,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        text=(response.choices[0].message.content or "").strip(),
    )


def print_result(result: Result) -> None:
    print(f"[{result.provider}] {result.model}")
    print(f"  latency: {result.latency_s:.2f}s")
    print(
        f"  tokens:  prompt={result.prompt_tokens} "
        f"completion={result.completion_tokens} total={result.total_tokens}"
    )
    print(f"  reply:   {result.text}\n")


def main() -> None:
    print(f"Prompt: {PROMPT}\n")

    for provider in PROVIDERS:
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            print(f"[{provider.name}] skipped: {provider.api_key_env} is not set\n")
            continue

        try:
            print_result(ask(provider, api_key, PROMPT))
        except OpenAIError as exc:
            print(f"[{provider.name}] failed: {exc}\n")


if __name__ == "__main__":
    main()
