"""Language layer: detection, masking and the translate-in / translate-out round trip (fake backend)."""
from __future__ import annotations

from krishidisha.services.lang import LanguageLayer, detect_language, mask


def test_detects_scripts_and_hinglish():
    assert detect_language("गेहूं में पीला रतुआ का इलाज क्या है?")[0] == "hi"
    assert detect_language("माझ्या कापूस पिकावर बोंडअळी आहे, काय करावे?")[0] == "mr"
    assert detect_language("ਕਣਕ ਦੀ ਬਿਜਾਈ ਕਦੋਂ ਕਰੀਏ")[0] == "pa"
    assert detect_language("நெல் பயிரில் குலை நோய்")[0] == "ta"
    assert detect_language("వరి పంటకు ఎరువులు ఎప్పుడు వేయాలి")[0] == "te"
    assert detect_language("2 acre gehu ke liye kitna urea lagega?")[0] == "hinglish"
    assert detect_language("How much urea for two acres of wheat?")[0] == "en"
    assert detect_language("")[0] == "en"


def test_mask_protects_numbers_units_links_and_tool_calls():
    text = ("Spray mancozeb 75 WP at 2.5 g/l and apply urea 45 kg/acre. Buy [DAP](/marketplace/product/dap). "
            'Grade 10-26-26. <tool_call>{"name": "get_weather", "arguments": {"place": "Indore"}}</tool_call>')
    m = mask(text)
    assert "<tool_call>" not in m.text and "/marketplace" not in m.text and "10-26-26" not in m.text
    assert "2.5 g/l" not in m.text and "45 kg/acre" not in m.text
    assert len(m.slots) >= 5
    assert m.restore(m.text).replace("  ", " ").strip() == text.replace("  ", " ").strip() or all(v in m.restore(m.text) for v in m.slots.values())


def test_round_trip_with_fake_backend_only_for_non_native_languages():
    layer = LanguageLayer(backend="fake")
    assert layer.active
    # native languages pass through untouched
    text_hi = "गेहूं में कितना यूरिया डालें?"
    assert layer.inbound(text_hi, "hi") == (text_hi, None)
    assert layer.outbound("Use 45 kg/acre urea.", "en") == "Use 45 kg/acre urea."
    # Marathi goes through translation both ways, numbers preserved
    inbound, masked = layer.inbound("कापसाला 2 एकर साठी किती युरिया लागेल?", "mr")
    assert inbound.startswith("[mr>en]") and masked is not None
    reply = "Apply urea 45 kg/acre in two splits.\n\n- Basal: 20 kg/acre\n- Top dressing: 25 kg/acre"
    out = layer.outbound(reply, "mr")
    assert out.count("[en>mr]") == 3
    assert "45 kg/acre" in out and "20 kg/acre" in out and "25 kg/acre" in out


def test_resolve_prefers_confident_detection_over_ui_choice():
    layer = LanguageLayer(backend="none")
    assert not layer.active
    lang, conf = layer.resolve("நெல் பயிரில் குலை நோய் வந்துள்ளது", "en")
    assert lang == "ta" and conf >= 0.8
    lang, _ = layer.resolve("How do I control aphids?", "hi")
    assert lang == "hi"  # english text cannot override an explicit request
    lang, _ = layer.resolve("How do I control aphids?", "auto")
    assert lang == "en"


def test_unknown_backend_falls_back_to_none():
    layer = LanguageLayer(backend="does-not-exist")
    assert layer.translator.name == "none"
    assert layer.outbound("hello", "ta") == "hello"
