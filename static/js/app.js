/* KrishiDisha shared front-end: chat widget, add-to-cart, small helpers. */
(function () {
  "use strict";
  const KD = window.KD || {};

  // ------------------------------------------------------------ helpers
  function renderMarkdown(text) {
    if (window.marked && window.DOMPurify) {
      return DOMPurify.sanitize(marked.parse(text || "", { breaks: true, gfm: true }));
    }
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
  }
  window.KDrenderMarkdown = renderMarkdown;

  function toast(message, kind) {
    let wrap = document.getElementById("kdToastWrap");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.id = "kdToastWrap";
      wrap.className = "toast-container position-fixed top-0 end-0 p-3";
      wrap.style.zIndex = 1080;
      document.body.appendChild(wrap);
    }
    const el = document.createElement("div");
    el.className = "toast align-items-center text-bg-" + (kind || "success") + " border-0";
    el.setAttribute("role", "status");
    el.innerHTML = '<div class="d-flex"><div class="toast-body">' + message + '</div>' +
      '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button></div>';
    wrap.appendChild(el);
    if (window.bootstrap) { new bootstrap.Toast(el, { delay: 3000 }).show(); }
    el.addEventListener("hidden.bs.toast", () => el.remove());
  }
  window.KDtoast = toast;

  // ------------------------------------------------------------ add to cart
  document.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("[data-add-to-cart]");
    if (!btn) return;
    ev.preventDefault();
    if (!KD.loggedIn) {
      window.location.href = KD.loginUrl + "?next=" + encodeURIComponent(window.location.pathname);
      return;
    }
    const qtyInput = btn.dataset.qtyInput ? document.querySelector(btn.dataset.qtyInput) : null;
    const quantity = qtyInput ? parseInt(qtyInput.value || "1", 10) : 1;
    btn.disabled = true;
    try {
      const r = await fetch(KD.cartAddUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ product_id: btn.dataset.addToCart, quantity })
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Could not add to cart");
      const badge = document.getElementById("cartCount");
      if (badge) { badge.textContent = data.cart_count; badge.hidden = !data.cart_count; }
      toast(data.message || "Added to cart", "success");
    } catch (e) {
      toast(e.message, "danger");
    } finally {
      btn.disabled = false;
    }
  });

  // ------------------------------------------------------------ chat widget
  const toggle = document.getElementById("kdChatToggle");
  const panel = document.getElementById("kdChatPanel");
  if (!toggle || !panel) return;
  const closeBtn = document.getElementById("kdChatClose");
  const form = document.getElementById("kdChatForm");
  const input = document.getElementById("kdChatInput");
  const imageInput = document.getElementById("kdChatImage");
  const messages = document.getElementById("kdChatMessages");
  const status = document.getElementById("kdChatStatus");
  let sessionKey = null;
  try { sessionKey = sessionStorage.getItem("kd_chat_session"); } catch (e) { /* ignore */ }

  function open() { panel.hidden = false; toggle.hidden = true; input.focus(); }
  function close() { panel.hidden = true; toggle.hidden = false; }
  toggle.addEventListener("click", open);
  closeBtn.addEventListener("click", close);

  function addMessage(role, html, meta, ref) {
    const div = document.createElement("div");
    div.className = "kd-msg " + (role === "user" ? "kd-msg-user" : "kd-msg-bot");
    div.innerHTML = html;
    messages.appendChild(div);
    if (meta || ref) {
      const m = document.createElement("div");
      m.className = "kd-msg-meta";
      m.textContent = meta || "";
      if (ref) {
        m.dataset.ref = ref;
        m.insertAdjacentHTML("beforeend",
          ' <button type="button" class="btn btn-sm btn-link p-0 ms-2 kd-rate-btn" data-rating="1"' +
          ' aria-label="This reply was helpful"><i class="bi bi-hand-thumbs-up"></i></button>' +
          ' <button type="button" class="btn btn-sm btn-link p-0 ms-1 kd-rate-btn" data-rating="-1"' +
          ' aria-label="This reply was not helpful"><i class="bi bi-hand-thumbs-down"></i></button>');
      }
      messages.appendChild(m);
    }
    messages.scrollTop = messages.scrollHeight;
    return div;
  }

  // -------------------------------------------------- widget reply feedback
  function rateReply(metaEl, rating, comment) {
    fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind: "chat", ref_id: metaEl.dataset.ref, rating: rating,
        comment: comment || null, language: KD.language || "en"
      })
    }).catch(() => { /* feedback is best-effort */ });
    metaEl.querySelectorAll(".kd-rate-btn").forEach((b) => { b.disabled = true; });
    const note = document.createElement("span");
    note.className = "ms-2 text-success small";
    note.textContent = "Thanks!";
    metaEl.appendChild(note);
  }

  messages.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".kd-rate-btn");
    if (!btn) return;
    const metaEl = btn.closest(".kd-msg-meta");
    if (!metaEl || !metaEl.dataset.ref) return;
    const rating = parseInt(btn.dataset.rating, 10);
    if (rating > 0) { rateReply(metaEl, 1); return; }
    if (metaEl.querySelector(".kd-feedback-comment")) return;
    const wrap = document.createElement("div");
    wrap.className = "kd-feedback-comment d-flex gap-1 mt-1";
    wrap.innerHTML = '<input type="text" class="form-control form-control-sm" maxlength="300"' +
      ' placeholder="What was wrong? (optional)" aria-label="What was wrong with this reply?">' +
      '<button type="button" class="btn btn-sm btn-outline-success kd-feedback-send">Send</button>';
    metaEl.appendChild(wrap);
    const box = wrap.querySelector("input");
    let sent = false;
    const done = () => {
      if (sent) return;
      sent = true;
      const text = box.value.trim();
      wrap.remove();
      rateReply(metaEl, -1, text);
    };
    box.focus();
    wrap.querySelector(".kd-feedback-send").addEventListener("click", done);
    box.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); done(); } });
    box.addEventListener("blur", () => setTimeout(done, 250));
  });

  imageInput.addEventListener("change", () => {
    if (imageInput.files && imageInput.files[0]) {
      status.textContent = "Photo attached: " + imageInput.files[0].name;
    }
  });

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    const file = imageInput.files && imageInput.files[0];
    if (!text && !file) return;

    let userHtml = "";
    if (file) {
      userHtml += '<img src="' + URL.createObjectURL(file) + '" alt="leaf photo">';
    }
    const esc = document.createElement("div"); esc.textContent = text; userHtml += esc.innerHTML;
    addMessage("user", userHtml || "(photo)");
    input.value = "";
    const typing = addMessage("bot", '<span class="kd-typing"><span></span><span></span><span></span></span>');
    status.textContent = "";

    try {
      let r;
      if (file) {
        const fd = new FormData();
        fd.append("message", text);
        fd.append("image", file);
        fd.append("language", KD.language || "en");
        if (sessionKey) fd.append("session_key", sessionKey);
        r = await fetch(KD.chatUrl, { method: "POST", body: fd });
      } else {
        r = await fetch(KD.chatUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_key: sessionKey, language: KD.language || "en" })
        });
      }
      const data = await r.json();
      typing.remove();
      if (!r.ok) { addMessage("bot", "Sorry: " + (data.error || "something went wrong")); return; }
      sessionKey = data.session_key;
      try { sessionStorage.setItem("kd_chat_session", sessionKey); } catch (e) { /* ignore */ }
      let html = renderMarkdown(data.reply);
      if (data.detection && data.detection.top) {
        const t = data.detection.top;
        html = '<div class="small text-success mb-1"><i class="bi bi-camera"></i> ' + t.name + " (" + Math.round(t.confidence * 100) + "%)</div>" + html;
      }
      const meta = (data.provider || "") + (data.tools_used && data.tools_used.length ? " · " + data.tools_used.join(", ") : "");
      const ref = sessionKey ? sessionKey + (data.message_index !== undefined ? "#" + data.message_index : "") : null;
      addMessage("bot", html, meta, ref);
    } catch (e) {
      typing.remove();
      addMessage("bot", "Network error. Please try again.");
    } finally {
      imageInput.value = "";
    }
  });
})();
