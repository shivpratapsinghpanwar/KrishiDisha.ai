"""KrishiDisha Agri-LLM assistant.

Providers
---------
* ``anthropic`` - Claude via the official SDK, with native tool use and vision.
* ``openai``    - any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, Ollama, LM Studio)
                  using function calling.
* ``rules``     - offline fallback that still runs the ML models, weather, prices and
                  knowledge search through intent detection. Always available.

Every provider shares the same tool registry (``tools.py``) and the same
retrieval step: the top knowledge-base passages for the question are attached
to the user turn so answers are grounded in the project's data.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

from .fallback_bot import RulesBot
from .tools import Tool, run_tool

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "hinglish": "Hinglish (Hindi written in Latin script)", "mr": "Marathi",
    "pa": "Punjabi", "gu": "Gujarati", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "bn": "Bengali",
    "ml": "Malayalam", "or": "Odia",
}

SYSTEM_PROMPT = """You are KrishiDisha Sahayak, an expert agricultural advisor for Indian farmers, built into the KrishiDisha platform (crop recommendation, fertilizer recommendation, disease detection from leaf photos, yield prediction, weather, mandi prices and an input marketplace).

How to work:
- Be practical and specific: doses in kg/acre or ml/litre, timings, product names, costs in INR. Prefer Indian units (acre, quintal, bigha when the farmer uses it).
- Use the tools whenever numbers, forecasts, prices, products or catalogue facts are needed instead of guessing. You may call several tools in one turn. If a tool needs inputs the farmer has not given (e.g. NPK values), ask for them briefly, and mention that a Soil Health Card or a soil test gives these values.
- Ground disease and pest advice in the knowledge base results and the disease detection output when an image was analysed. Always mention prevention as well as cure, give an organic/IPM option first when reasonable, and remind about safety gear and pre-harvest interval for chemical sprays.
- When you recommend a fertilizer, fungicide, seed or tool that the marketplace sells, name it and give its link as a markdown link (path starts with /marketplace/product/).
- Reply in the farmer's language: if the message is in Hindi (Devanagari) answer in Hindi; if it is Hinglish answer in Hinglish; otherwise use the requested language. Keep sentences short and use bullet points and small tables for schedules. Do not use headings larger than ###.
- If a question is outside agriculture, politely steer back to farming topics. Never invent government scheme figures; use the scheme tool.
- Keep answers focused (usually under 250 words) unless a detailed plan is requested."""


class AgriAssistant:
    def __init__(self, config: Any, tools: list[Tool], kb, detector=None):
        self.cfg = config
        self.tools = tools
        self.kb = kb
        self.detector = detector
        self.rules = RulesBot(tools, kb)
        self.provider = self._choose_provider()
        self._anthropic = None
        self._openai = None
        log.info("Agri assistant provider: %s", self.provider)

    # ------------------------------------------------------------ provider
    def _choose_provider(self) -> str:
        pref = (getattr(self.cfg, "LLM_PROVIDER", "auto") or "auto").lower()
        if pref in {"anthropic", "openai", "rules"}:
            return pref
        if getattr(self.cfg, "ANTHROPIC_API_KEY", None):
            return "anthropic"
        if getattr(self.cfg, "OPENAI_API_KEY", None) or getattr(self.cfg, "OPENAI_BASE_URL", None):
            return "openai"
        return "rules"

    @property
    def anthropic_client(self):
        if self._anthropic is None:
            import anthropic

            self._anthropic = anthropic.Anthropic(api_key=self.cfg.ANTHROPIC_API_KEY, max_retries=2, timeout=90.0)
        return self._anthropic

    @property
    def openai_client(self):
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI(api_key=self.cfg.OPENAI_API_KEY or "ollama", base_url=self.cfg.OPENAI_BASE_URL,
                                  timeout=90.0, max_retries=2)
        return self._openai

    def describe(self) -> dict[str, Any]:
        model = {"anthropic": self.cfg.ANTHROPIC_MODEL, "openai": self.cfg.OPENAI_MODEL, "rules": "offline-rules"}
        return {"provider": self.provider, "model": model.get(self.provider), "tools": [t.name for t in self.tools]}

    # ---------------------------------------------------------------- chat
    def chat(self, message: str, history: list[dict[str, str]] | None = None, farmer_context: str | None = None,
             language: str = "en", image_bytes: bytes | None = None, image_mime: str = "image/jpeg") -> dict[str, Any]:
        history = [h for h in (history or []) if h.get("role") in {"user", "assistant"} and h.get("content")]
        history = history[-self.cfg.CHAT_HISTORY_TURNS:]

        # 1. Optional image -> run disease model first (works for every provider)
        vision_note = None
        detection = None
        if image_bytes and self.detector is not None:
            try:
                from PIL import Image

                detection = self.detector.predict(Image.open(io.BytesIO(image_bytes)))
                if detection.get("available"):
                    top = detection["top"]
                    if not detection.get("is_plant", True):
                        vision_note = (
                            f"[The KrishiDisha disease model could not recognise a plant leaf in the photo "
                            f"(top guess {top['name']} at {top['confidence'] * 100:.0f}%). Tell the farmer to retake it: one leaf "
                            f"filling the frame, daylight, plain background. Do not diagnose from this photo.]"
                        )
                    else:
                        info = self.kb.disease_by_label(top["label"]) or {}
                        caveat = ""
                        if detection.get("uncertain"):
                            caveat = " The model is NOT confident: present this as a possibility, ask about symptoms, and advise confirming with a KVK."
                        elif detection.get("crop_tier") == "C":
                            caveat = " This crop has limited training data (experimental); advise confirming with an expert."
                        vision_note = (
                            f"[Leaf image analysed by the KrishiDisha disease model ({detection['model']}): "
                            f"top prediction {top['name']} with {top['confidence'] * 100:.1f}% confidence; "
                            f"other candidates: {', '.join(p['name'] + ' ' + str(round(p['confidence'] * 100, 1)) + '%' for p in detection['predictions'][1:])}.{caveat} "
                            f"Catalogue guidance: {info.get('description', '')[:600]} Steps: {info.get('prevention', '')[:600]}]"
                        )
            except Exception as exc:  # noqa: BLE001
                log.warning("Image analysis failed: %s", exc)

        # 2. Retrieval: attach grounding passages to the user turn
        passages = self.kb.search(message, k=3)
        grounding = ""
        if passages:
            grounding = "\n\n[Knowledge base passages that may help (cite them when used):\n" + "\n".join(
                f"- ({p['source']}) {p['title']}: {p['text'][:500]}" for p in passages) + "]"

        lang_name = LANGUAGE_NAMES.get(language, language)
        user_turn = message
        if farmer_context:
            user_turn = f"[Farmer profile: {farmer_context}]\n" + user_turn
        if vision_note:
            user_turn += "\n\n" + vision_note
        user_turn += grounding
        user_turn += f"\n\n[Preferred reply language: {lang_name}]"

        try:
            if self.provider == "anthropic":
                return self._chat_anthropic(user_turn, history, image_bytes, image_mime, passages, detection)
            if self.provider == "openai":
                return self._chat_openai(user_turn, history, image_bytes, image_mime, passages, detection)
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM provider %s failed; using offline fallback", self.provider)
            result = self.rules.reply(message, language=language, detection=detection, farmer_context=farmer_context)
            result["provider"] = "rules"
            result["fallback_reason"] = str(exc)[:300]
            return result
        result = self.rules.reply(message, language=language, detection=detection, farmer_context=farmer_context)
        result["provider"] = "rules"
        return result

    # ------------------------------------------------------------ anthropic
    def _chat_anthropic(self, user_turn, history, image_bytes, image_mime, passages, detection):
        import anthropic

        client = self.anthropic_client
        content: list[dict[str, Any]] = []
        if image_bytes:
            content.append({"type": "image", "source": {"type": "base64", "media_type": image_mime,
                                                        "data": base64.standard_b64encode(image_bytes).decode()}})
        content.append({"type": "text", "text": user_turn})
        messages: list[dict[str, Any]] = [{"role": h["role"], "content": h["content"]} for h in history]
        messages.append({"role": "user", "content": content})
        tool_defs = [t.anthropic() for t in self.tools]
        tools_used: list[str] = []
        model = self.cfg.ANTHROPIC_MODEL
        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=4096,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=tool_defs,
        )
        if _supports_effort(model):
            kwargs["output_config"] = {"effort": "medium"}

        response = None
        for _ in range(self.cfg.LLM_MAX_TOOL_ROUNDS + 1):
            try:
                response = client.messages.create(messages=messages, **kwargs)
            except anthropic.BadRequestError as exc:
                if "output_config" in str(exc) or "effort" in str(exc):
                    kwargs.pop("output_config", None)
                    response = client.messages.create(messages=messages, **kwargs)
                else:
                    raise
            if response.stop_reason == "refusal":
                return {"reply": "I can't help with that request. Please ask me about crops, soil, weather, "
                                 "diseases, fertilizers or market prices.", "tools_used": tools_used,
                        "provider": "anthropic", "sources": [], "detection": detection}
            if response.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    tools_used.append(block.name)
                    out, is_err = run_tool(self.tools, block.name, dict(block.input))
                    item = {"type": "tool_result", "tool_use_id": block.id, "content": out}
                    if is_err:
                        item["is_error"] = True
                    results.append(item)
            messages.append({"role": "user", "content": results})
        text = "".join(b.text for b in response.content if b.type == "text") if response else ""
        if not text:
            text = "I could not produce an answer right now. Please try again."
        return {"reply": text, "tools_used": tools_used, "provider": "anthropic", "model": model,
                "sources": [p["title"] for p in passages], "detection": detection}

    # --------------------------------------------------------------- openai
    def _chat_openai(self, user_turn, history, image_bytes, image_mime, passages, detection):
        client = self.openai_client
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": h["role"], "content": h["content"]} for h in history]
        if image_bytes:
            b64 = base64.standard_b64encode(image_bytes).decode()
            messages.append({"role": "user", "content": [
                {"type": "text", "text": user_turn},
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}"}}]})
        else:
            messages.append({"role": "user", "content": user_turn})
        tool_defs = [t.openai() for t in self.tools]
        tools_used: list[str] = []
        model = self.cfg.OPENAI_MODEL
        msg = None
        for _ in range(self.cfg.LLM_MAX_TOOL_ROUNDS + 1):
            try:
                resp = client.chat.completions.create(model=model, messages=messages, tools=tool_defs,
                                                      tool_choice="auto", temperature=0.3)
            except Exception as exc:  # noqa: BLE001 - some local models reject tools/images
                if image_bytes and "image" in str(exc).lower():
                    messages[-1] = {"role": "user", "content": user_turn}
                    image_bytes = None
                    continue
                if "tool" in str(exc).lower() and tool_defs:
                    tool_defs = []
                    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
                else:
                    raise
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in calls]})
            for c in calls:
                tools_used.append(c.function.name)
                try:
                    args = json.loads(c.function.arguments or "{}")
                except ValueError:
                    args = {}
                out, _ = run_tool(self.tools, c.function.name, args)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
        text = (msg.content if msg else "") or "I could not produce an answer right now. Please try again."
        return {"reply": text, "tools_used": tools_used, "provider": "openai", "model": model,
                "sources": [p["title"] for p in passages], "detection": detection}


def _supports_effort(model: str) -> bool:
    m = model.lower()
    return any(k in m for k in ("opus-5", "sonnet-5", "fable", "opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6"))


def strip_markdown(text: str) -> str:
    """Plain-text version for TTS / SMS channels."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`#>]+", "", text)
    return text.strip()
