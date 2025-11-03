from __future__ import annotations

import os
from typing import Dict, List, Tuple
from proyecto_cero.settings import Settings


def build_prompt(query: str, docs: List[str], metas: List[Dict]) -> str:
    header = (
        "Eres un asistente que responde en español de forma concisa y accionable.\n"
        "Usa exclusivamente la información del CONTEXTO. Si algo no está en el contexto, di que no está disponible.\n"
        "Incluye pasos claros y, al final, lista breves referencias a las fuentes.\n\n"
    )
    context_parts = []
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        src = f"{m.get('filename','')}#p{m.get('page','')} c{m.get('chunk_index','')}"
        context_parts.append(f"[Fuente {i}: {src}]\n{d}")
    context = "\n\n".join(context_parts)
    return f"{header}PREGUNTA: {query}\n\nCONTEXTO:\n{context}\n\nRESPUESTA:"  # noqa: E501


def _gen_openai(prompt: str, model: str = None) -> str:
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un asistente útil en español."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _gen_ollama(prompt: str, model: str = None) -> str:
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    import ollama  # lazy import

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un asistente útil en español."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2},
    )
    return (resp.get("message", {}) or {}).get("content", "").strip()


def _gen_groq(prompt: str, model: str = None) -> str:
    model = model or os.getenv("GROQ_MODEL", "llama3-8b-8192")
    from groq import Groq  # lazy import

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un asistente útil en español. Solo contesta con lo que hay en los PDF de documentacion , solo si la informacion esta en los documentos."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _gen_xai(prompt: str, model: str = None) -> str:
    # xAI Grok API: OpenAI-compatible; use base_url
    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    model = model or os.getenv("XAI_MODEL", "grok-2-latest")
    from openai import OpenAI  # reuse OpenAI SDK with custom base_url

    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un asistente útil en español."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _gen_bedrock(prompt: str, model: str, region: str) -> str:
    import boto3  # lazy import

    client = boto3.client("bedrock-runtime", region_name=region)
    # Por defecto asumimos modelo Anthropic Claude 3.x en Bedrock
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 700,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    resp = client.invoke_model(modelId=model, body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    # Extraer texto según formato Claude
    parts = payload.get("content", [])
    for p in parts:
        if p.get("type") == "text" and "text" in p:
            return p["text"].strip()
    # Fallback
    return json.dumps(payload)


def answer_with_llm(query: str, docs: List[str], metas: List[Dict], settings: Settings | None = None) -> Tuple[str, str]:
    """Devuelve (texto_respuesta, proveedor_usado). Selecciona proveedor por entorno.

    Variables de entorno admitidas:
    - LLM_PROVIDER: "openai" | "ollama"
    - OPENAI_API_KEY, OPENAI_MODEL
    - OLLAMA_MODEL
    """
    prompt = build_prompt(query, docs, metas)
    s = settings or Settings.from_env()
    provider = (s.llm_provider or ("openai" if os.getenv("OPENAI_API_KEY") else "ollama")).lower()
    if provider == "openai":
        return _gen_openai(prompt, model=s.openai_model), "openai"
    if provider == "ollama":
        return _gen_ollama(prompt, model=s.ollama_model), "ollama"
    if provider == "groq":
        return _gen_groq(prompt, model=s.groq_model), "groq"
    if provider == "xai":
        return _gen_xai(prompt), "xai"
    if provider == "bedrock":
        return _gen_bedrock(prompt, model=s.bedrock_model, region=s.bedrock_region), "bedrock"
    # Fallback: intenta openai si hay API key; si no, ollama
    if os.getenv("OPENAI_API_KEY"):
        return _gen_openai(prompt, model=s.openai_model), "openai"
    if os.getenv("GROQ_API_KEY"):
        return _gen_groq(prompt, model=s.groq_model), "groq"
    return _gen_ollama(prompt, model=s.ollama_model), "ollama"
