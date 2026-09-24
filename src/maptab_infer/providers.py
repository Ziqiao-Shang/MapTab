from __future__ import annotations

import base64
import io
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    reasoning: str | None = None


class Provider(Protocol):
    def generate(self, content: list[dict[str, str]]) -> GenerationResult:
        ...


def _image_data_url(
    path: str,
    *,
    max_pixels: int,
    jpeg_quality: int = 95,
) -> str:
    source = Path(path)
    try:
        from PIL import Image

        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            pixels = width * height
            if max_pixels > 0 and pixels > max_pixels:
                scale = (max_pixels / pixels) ** 0.5
                image = image.resize(
                    (
                        max(1, int(width * scale)),
                        max(1, int(height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality)
            payload = buffer.getvalue()
        media_type = "image/jpeg"
    except ImportError:
        payload = source.read_bytes()
        media_type = mimetypes.guess_type(source.name)[0] or "image/png"
    return (
        f"data:{media_type};base64,"
        f"{base64.b64encode(payload).decode('ascii')}"
    )


def _openai_content(
    content: list[dict[str, str]],
    *,
    max_pixels: int,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for part in content:
        if part["type"] == "text":
            payload.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_path":
            payload.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(
                            part["path"],
                            max_pixels=max_pixels,
                        )
                    },
                }
            )
        else:
            raise ValueError(f"unknown content type: {part['type']}")
    return payload


class OpenAICompatibleProvider:
    """Hosted API or an OpenAI-compatible local serving endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
        max_pixels: int,
        max_retries: int,
        retry_backoff: float,
        timeout: float | None,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_pixels = max_pixels
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def generate(self, content: list[dict[str, str]]) -> GenerationResult:
        payload = _openai_content(content, max_pixels=self.max_pixels)
        delay = self.retry_backoff
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": payload}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )
                message = result.choices[0].message
                reasoning = getattr(message, "reasoning_content", None)
                return GenerationResult(message.content or "", reasoning)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(delay)
                delay *= 1.6
        assert last_error is not None
        raise last_error


class VLLMProvider:
    """In-process local inference matching MapTab-main's vLLM workflow."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
        max_pixels: int,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            disable_mm_preprocessor_cache=True,
        )
        self.sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        self.max_pixels = max_pixels

    def generate(self, content: list[dict[str, str]]) -> GenerationResult:
        messages = [
            {
                "role": "user",
                "content": _openai_content(
                    content,
                    max_pixels=self.max_pixels,
                ),
            }
        ]
        output = self.llm.chat(
            messages,
            sampling_params=self.sampling,
            use_tqdm=False,
        )
        text = output[0].outputs[0].text.strip()
        return GenerationResult(text)
