/* MF FAQ Assistant — front-end controller.
 *
 * The design mockup was static markup. This wires the same markup to the
 * FastAPI backend in server.py, which is a thin JSON adapter over
 * rag.pipeline.answer_question(). Nothing here decides what an answer says;
 * it renders what the pipeline returns, and it never rewrites the answer text.
 *
 * Rules that come from the pipeline rather than from taste:
 *
 *   - A refusal is shown as a refusal. The backend already refused it, so the
 *     UI marks the turn and does not dress it up as an answer.
 *   - The retrieved chunks are shown on demand. Same disclosure the Streamlit
 *     "Sources" expander gave, so "where did this come from" stays answerable.
 *   - The echo of the user's question is the redacted string the backend
 *     returned. A PAN is replaced with a placeholder before it ever reaches
 *     this code, and nothing here logs the raw text either.
 */

(function () {
  "use strict";

  var THEME_KEY = "mffaq.theme";
  var SESSION_KEY = "mffaq.session";
  var TRANSCRIPT_KEY = "mffaq.transcript";

  /* Refusal classification -> the pill the design shows above the text. */
  var MANDATE = {
    refuse_advice: {
      label: "Objective Mandate",
      note: "No Investment Advisory (SEBI IA Reg. Compliance)",
      icon: "info"
    },
    refuse_pii: {
      label: "Personal Data",
      note: "Identifiers are never stored or echoed",
      icon: "shield_lock"
    },
    refuse_offtopic: {
      label: "Out Of Scope",
      note: "Only the schemes in this corpus are covered",
      icon: "filter_alt_off"
    },
    refuse_dangling: {
      label: "Missing Reference",
      note: "Ask a full question rather than a follow-up reference",
      icon: "help"
    },
    no_context: {
      label: "Sources Incomplete",
      note: "Not covered by the retrieved official documents",
      icon: "search_off"
    }
  };

  var SUGGESTION_ICONS = ["search", "lock_clock", "speed", "verified", "receipt_long"];

  var el = {
    thread: document.getElementById("thread"),
    suggestions: document.getElementById("suggestions"),
    suggestionsRow: document.getElementById("suggestions-row"),
    input: document.getElementById("question-input"),
    send: document.getElementById("send-btn"),
    historyList: document.getElementById("history-list"),
    sourceList: document.getElementById("source-list"),
    sidebar: document.getElementById("sidebar"),
    sidebarToggle: document.getElementById("sidebar-toggle"),
    newInquiry: document.getElementById("new-inquiry"),
    themeToggle: document.getElementById("theme-toggle"),
    themeToggleIcon: document.getElementById("theme-toggle-icon"),
    helpBtn: document.getElementById("help-btn"),
    helpSheet: document.getElementById("help-sheet"),
    helpBody: document.getElementById("help-body"),
    banner: document.getElementById("banner-text"),
    footerNote: document.getElementById("footer-note")
  };

  var tpl = {
    empty: document.getElementById("tpl-empty"),
    user: document.getElementById("tpl-user"),
    assistant: document.getElementById("tpl-assistant"),
    pending: document.getElementById("tpl-pending"),
    error: document.getElementById("tpl-error"),
    chunk: document.getElementById("tpl-chunk")
  };

  var state = {
    sessionId: readStore(SESSION_KEY, ""),
    transcript: readStore(TRANSCRIPT_KEY, []),
    busy: false,
    meta: null,
    turnSeq: 0
  };

  /* --- storage ---------------------------------------------------------- */

  function readStore(key, fallback) {
    try {
      var raw = window.sessionStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (err) {
      return fallback;
    }
  }

  function writeStore(key, value) {
    try {
      window.sessionStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
      /* Private-mode or quota. The app still works, it just forgets on reload. */
    }
  }

  function saveTranscript() {
    writeStore(TRANSCRIPT_KEY, state.transcript);
  }

  /* --- text helpers ----------------------------------------------------- */

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (chr) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[chr];
    });
  }

  /* A deliberately small inline renderer: bold, italic, code, and links. The
   * pipeline's answers are at most three sentences, so a full markdown
   * parser would be dead weight — but the text is model output, so it is
   * escaped first and only these five patterns are re-inserted. */
  function renderMarkdown(text) {
    var escaped = escapeHtml(text);
    var html = escaped
      .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/`([^`\n]+)`/g, "<code>$1</code>");
    return html.split(/\n{2,}/).map(function (block) {
      var trimmed = block.trim();
      return trimmed ? "<p>" + trimmed.replace(/\n/g, "<br>") + "</p>" : "";
    }).join("");
  }

  function hostOf(url) {
    if (!url) {
      return "";
    }
    try {
      return new URL(url).host.replace(/^www\./, "");
    } catch (err) {
      return url;
    }
  }

  function pathOf(url) {
    if (!url) {
      return "";
    }
    try {
      var parsed = new URL(url);
      return (parsed.hostname.replace(/^www\./, "") + parsed.pathname).replace(/\/$/, "");
    } catch (err) {
      return url;
    }
  }

  function scrollToLatest() {
    window.requestAnimationFrame(function () {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
      });
    });
  }

  function setBusy(busy) {
    state.busy = busy;
    el.send.disabled = busy;
    el.input.disabled = busy;
    Array.prototype.forEach.call(
      el.suggestionsRow.querySelectorAll(".suggestion-chip"),
      function (chip) { chip.disabled = busy; }
    );
  }

  /* --- rendering: user turn -------------------------------------------- */

  function renderEmptyState() {
    el.thread.textContent = "";
    el.thread.appendChild(tpl.empty.content.firstElementChild.cloneNode(true));
  }

  function renderUser(question, turnId) {
    var node = tpl.user.content.firstElementChild.cloneNode(true);
    node.id = turnId;
    node.querySelector(".user-text").textContent = question;
    el.thread.appendChild(node);
  }

  /* --- rendering: assistant turn --------------------------------------- */

  function renderAssistant(turn) {
    var node = tpl.assistant.content.firstElementChild.cloneNode(true);
    node.id = turn.id;

    var verdict = turn.verdict || {};
    var mandate = MANDATE[verdict.action];
    var isRefusal = Boolean(turn.answer.refused) || Boolean(mandate);

    var avatar = node.querySelector(".avatar-assistant");
    var glyph = avatar.querySelector(".material-symbols-outlined");
    if (isRefusal) {
      avatar.classList.add("is-secondary");
      glyph.textContent = mandate ? mandate.icon : "info";
    } else {
      glyph.textContent = "balance";
    }

    /* Refusal banner. Suppressed on a plain answer so the bubble opens the
     * way the mockup does — with the prose. */
    if (mandate) {
      var mandateRow = node.querySelector(".mandate-row");
      mandateRow.hidden = false;
      mandateRow.querySelector(".pill-mandate").textContent = mandate.label;
      mandateRow.querySelector(".mandate-note").textContent = mandate.note;
    }

    if (turn.rewritten && turn.resolved_question) {
      var rewrite = node.querySelector(".rewrite-note");
      rewrite.hidden = false;
      rewrite.textContent = "interpreted as: " + turn.resolved_question;
    }

    node.querySelector(".answer-body").innerHTML = renderMarkdown(turn.answer.text);

    renderWarnings(node, turn.answer.warnings);
    renderSourceCards(node, turn);
    renderChunkPanel(node, turn.chunks);

    var citation = node.querySelector(".citation");
    if (!turn.answer.refused) {
      citation.textContent = citationText(turn);
    } else {
      node.querySelector(".answer-footer").classList.add("footer-end");
      citation.remove();
    }

    wireActions(node, turn);
    el.thread.appendChild(node);
  }

  function citationText(turn) {
    var parts = [];
    var top = (turn.chunks || [])[0];
    if (top && top.citation_label) {
      parts.push(top.citation_label);
    }
    if (turn.stage) {
      parts.push(turn.stage);
    }
    if (typeof turn.latency_s === "number") {
      parts.push(turn.latency_s.toFixed(2) + "s");
    }
    return parts.join(" · ");
  }

  function renderWarnings(node, warnings) {
    var box = node.querySelector(".warnings");
    if (!warnings || !warnings.length) {
      return;
    }
    box.hidden = false;
    warnings.forEach(function (text) {
      var row = document.createElement("div");
      row.className = "warning-row";
      row.innerHTML = '<span class="material-symbols-outlined ms-15" aria-hidden="true">' +
        "warning</span><span></span>";
      row.lastElementChild.textContent = text;
      box.appendChild(row);
    });
  }

  function sourceCard(spec) {
    var card = document.createElement("div");
    card.className = "source-card";

    var main = document.createElement("div");
    main.className = "source-main";

    var glyph = document.createElement("span");
    glyph.className = "material-symbols-outlined ms-20 " + spec.tone;
    glyph.setAttribute("aria-hidden", "true");
    glyph.textContent = spec.icon;

    var copy = document.createElement("div");
    copy.className = "source-copy";

    var title = document.createElement("div");
    title.className = "source-title truncate";
    title.textContent = spec.title;
    title.title = spec.title;

    var meta = document.createElement("div");
    meta.className = "source-meta";

    var url = document.createElement("a");
    url.className = "source-url";
    url.href = spec.url;
    url.target = "_blank";
    url.rel = "noopener noreferrer";
    url.textContent = spec.urlLabel;
    url.title = spec.url;

    var bullet = document.createElement("span");
    bullet.textContent = "•";

    var stamp = document.createElement("span");
    stamp.className = "source-stamp";
    stamp.textContent = spec.stamp;

    meta.append(url, bullet, stamp);

    copy.append(title, meta);
    main.append(glyph, copy);

    var badge = document.createElement("div");
    badge.className = "source-badge" + (spec.tone === "icon-secondary" ? " is-secondary" : "");
    badge.innerHTML = '<span class="material-symbols-outlined ms-15" aria-hidden="true"></span><span></span>';
    badge.firstElementChild.textContent = spec.badgeIcon;
    badge.lastElementChild.textContent = spec.badge;

    card.append(main, badge);
    return card;
  }

  function renderSourceCards(node, turn) {
    var box = node.querySelector(".source-cards");
    var answer = turn.answer;
    var top = (turn.chunks || [])[0];

    if (answer.source_url) {
      box.appendChild(sourceCard({
        icon: "article",
        tone: "icon-primary",
        title: top
          ? top.scheme_short + " — " + top.doc_type
          : "Official scheme document",
        url: answer.source_url,
        urlLabel: pathOf(answer.source_url) || hostOf(answer.source_url),
        stamp: answer.freshness ? "Updated " + answer.freshness : hostOf(answer.source_url),
        badgeIcon: "verified_user",
        badge: "Official Source"
      }));
    }

    if (answer.link) {
      box.appendChild(sourceCard({
        icon: "school",
        tone: "icon-secondary",
        title: "Investor education resource",
        url: answer.link,
        urlLabel: pathOf(answer.link) || hostOf(answer.link),
        stamp: "Statutory Education Portal",
        badgeIcon: "menu_book",
        badge: "Educational Resource"
      }));
    }
  }

  /* The Streamlit UI showed what was actually retrieved under every answer.
   * Same information, same distance semantics, folded away by default. */
  function renderChunkPanel(node, chunks) {
    if (!chunks || !chunks.length) {
      return;
    }
    var panel = node.querySelector(".chunk-panel");
    var toggle = panel.querySelector(".chunk-toggle");
    var list = panel.querySelector(".chunk-list");

    panel.querySelector(".chunk-toggle-label").textContent =
      "Sources (" + chunks.length + " chunks retrieved)";

    toggle.setAttribute("aria-expanded", "false");
    toggle.addEventListener("click", function () {
      var open = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", open ? "false" : "true");
      list.hidden = open;
    });

    chunks.forEach(function (chunk) {
      var row = tpl.chunk.content.firstElementChild.cloneNode(true);
      row.querySelector(".chunk-title").textContent =
        chunk.scheme_short + " · " + chunk.section;
      row.querySelector(".chunk-distance").textContent =
        "distance " + chunk.distance.toFixed(3);
      var urlNode = row.querySelector(".chunk-url");
      if (chunk.source_url) {
        var anchor = document.createElement("a");
        anchor.href = chunk.source_url;
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
        anchor.textContent = chunk.source_url;
        urlNode.appendChild(anchor);
      }
      var body = String(chunk.text || "").replace(/\s+/g, " ").trim();
      row.querySelector(".chunk-text").textContent =
        body.length > 280 ? body.slice(0, 280) + "…" : body;
      list.appendChild(row);
    });
  }

  function wireActions(node, turn) {
    node.querySelectorAll(".action-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var act = btn.dataset.act;
        if (act === "copy") {
          copyText(turn.answer.text, btn);
          return;
        }
        if (act === "up" || act === "down") {
          var group = btn.parentElement;
          var wasActive = btn.classList.contains("is-active");
          group.querySelectorAll(".action-btn").forEach(function (other) {
            other.classList.remove("is-active");
          });
          if (!wasActive) {
            btn.classList.add("is-active");
          }
          return;
        }
        if (act === "retry") {
          ask(turn.question);
        }
      });
    });
  }

  function copyText(text, btn) {
    var done = function () {
      var glyph = btn.querySelector(".material-symbols-outlined");
      var original = glyph.textContent;
      glyph.textContent = "check";
      btn.classList.add("is-active");
      window.setTimeout(function () {
        glyph.textContent = original;
        btn.classList.remove("is-active");
      }, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () {});
      return;
    }
    var scratch = document.createElement("textarea");
    scratch.value = text;
    document.body.appendChild(scratch);
    scratch.select();
    try {
      document.execCommand("copy");
      done();
    } catch (err) {
      /* Clipboard unavailable; nothing to recover. */
    }
    document.body.removeChild(scratch);
  }

  function renderError(message, question) {
    var node = tpl.error.content.firstElementChild.cloneNode(true);
    node.querySelector(".error-text").textContent = message;
    wireActions(node, { question: question });
    el.thread.appendChild(node);
  }

  function renderPending() {
    var node = tpl.pending.content.firstElementChild.cloneNode(true);
    el.thread.appendChild(node);
    return node;
  }

  function nextTurnId() {
    state.turnSeq += 1;
    return "turn-" + Date.now() + "-" + state.turnSeq;
  }

  /* --- shell chrome ----------------------------------------------------- */

  function renderSuggestions() {
    var examples = (state.meta && state.meta.examples) || [];
    el.suggestionsRow.textContent = "";
    examples.forEach(function (question, index) {
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "suggestion-chip";
      chip.innerHTML = '<span class="material-symbols-outlined ms-15 icon-primary" ' +
        'aria-hidden="true"></span><span></span>';
      chip.firstElementChild.textContent = SUGGESTION_ICONS[index % SUGGESTION_ICONS.length];
      chip.lastElementChild.textContent = question;
      chip.addEventListener("click", function () { ask(question); });
      el.suggestionsRow.appendChild(chip);
    });
  }

  function renderHistory() {
    el.historyList.textContent = "";
    var entries = state.transcript.filter(function (turn) {
      return turn.role === "user";
    });
    if (!entries.length) {
      var empty = document.createElement("span");
      empty.className = "history-empty";
      empty.textContent = "No inquiries yet.";
      el.historyList.appendChild(empty);
      return;
    }
    /* The active marker tracks the newest *inquiry*, which is the last user
     * turn - not the last entry in the transcript, which is its answer. */
    var newest = entries[entries.length - 1].id;
    entries.slice().reverse().forEach(function (entry) {
      var item = document.createElement("button");
      item.type = "button";
      item.className = "history-item";
      item.textContent = entry.question;
      item.title = entry.question;
      if (entry.id === newest) {
        item.setAttribute("aria-current", "true");
      }
      item.addEventListener("click", function () {
        var target = document.getElementById(entry.id);
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
      el.historyList.appendChild(item);
    });
  }

  function renderSources() {
    var sources = (state.meta && state.meta.official_sources) || [];
    el.sourceList.textContent = "";
    sources.forEach(function (label) {
      var row = document.createElement("li");
      row.innerHTML = '<span class="material-symbols-outlined ms-15 check-muted" ' +
        'aria-hidden="true">check_circle</span><span></span>';
      row.lastElementChild.textContent = label;
      el.sourceList.appendChild(row);
    });
  }

  function renderStartupProblems() {
    var problems = []
      .concat((state.meta && state.meta.missing_env) || [])
      .concat((state.meta && state.meta.boot_errors) || []);
    if (!problems.length) {
      return;
    }
    var banner = document.createElement("div");
    banner.className = "startup-error";
    banner.textContent = problems.length === 1
      ? "Startup problem: " + problems[0]
      : "Startup problems: " + problems.join("; ");
    el.suggestions.parentNode.insertBefore(banner, el.suggestions);
    el.send.disabled = true;
  }

  function renderHelp() {
    var meta = state.meta || {};
    var rows = [
      ["What this answers",
        "Expense ratios, exit loads, minimum SIP amounts, benchmarks, AUM, " +
        "riskometer readings and scheme facts for the five HDFC Direct-Growth " +
        "schemes in this demo's corpus."],
      ["What it refuses",
        "Anything asking for a recommendation, a comparison of which fund is " +
        "'best', a performance forecast, or a personal identifier. Refusals are " +
        "mandated by the assistant's facts-only scope, not by a model preference."],
      ["Where answers come from",
        meta.corpus_note || ""],
      ["How memory works",
        "The last ten messages stay in this browser tab to resolve follow-ups " +
        "like \"and its exit load?\". \"New inquiry\" clears them. Nothing is " +
        "written to disk on the server and nothing survives the tab closing."]
    ];
    el.helpBody.textContent = "";
    rows.forEach(function (row) {
      var heading = document.createElement("h3");
      heading.textContent = row[0];
      var body = document.createElement("p");
      body.textContent = row[1];
      el.helpBody.append(heading, body);
    });
  }

  function applyTheme(theme) {
    var dark = theme === "dark";
    document.body.classList.toggle("theme-dark", dark);
    document.body.classList.toggle("theme-light", !dark);
    el.themeToggleIcon.textContent = dark ? "light_mode" : "dark_mode";
    try {
      window.localStorage.setItem(THEME_KEY, theme);
    } catch (err) {
      /* Storage blocked; the toggle still works for this page view. */
    }
  }

  function initTheme() {
    var stored = null;
    try {
      stored = window.localStorage.getItem(THEME_KEY);
    } catch (err) {
      stored = null;
    }
    var theme = stored || (window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    applyTheme(theme);
  }

  /* --- asking ----------------------------------------------------------- */

  function ask(question) {
    question = String(question || "").trim();
    if (!question || state.busy) {
      return;
    }

    var emptyState = document.querySelector(".empty-state");
    if (emptyState) {
      emptyState.remove();
    }

    var userId = nextTurnId();
    renderUser(question, userId);
    state.transcript.push({ id: userId, role: "user", question: question });
    saveTranscript();
    renderHistory();

    el.input.value = "";
    el.send.disabled = true;
    setBusy(true);
    var pending = renderPending();
    scrollToLatest();

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, session_id: state.sessionId })
    })
      .then(function (response) {
        return response.json()
          .catch(function () { return {}; })
          .then(function (payload) {
            if (!response.ok) {
              throw new Error(payload.detail || "The assistant could not answer.");
            }
            return payload;
          });
      })
      .then(function (payload) {
        pending.remove();
        state.sessionId = payload.session_id || state.sessionId;
        writeStore(SESSION_KEY, state.sessionId);

        var assistantId = nextTurnId();
        var turn = {
          id: assistantId,
          role: "assistant",
          question: payload.question,
          answer: payload.answer,
          chunks: payload.chunks || [],
          verdict: payload.verdict || {},
          stage: payload.stage,
          latency_s: payload.latency_s,
          rewritten: payload.rewritten,
          resolved_question: payload.resolved_question
        };
        state.transcript.push(turn);
        saveTranscript();
        renderAssistant(turn);
        renderHistory();
        scrollToLatest();
      })
      .catch(function (err) {
        pending.remove();
        var message = err && err.message ? err.message : "The assistant could not answer.";
        /* Drop the question again. A turn that never produced an answer would
         * otherwise be replayed on refresh as a question with no reply, and the
         * error card carrying the retry is transient by design. */
        var index = state.transcript.length - 1;
        if (index >= 0 && state.transcript[index].role === "user") {
          state.transcript.pop();
        }
        var failed = document.getElementById(userId);
        if (failed) {
          failed.remove();
        }
        saveTranscript();
        renderHistory();
        renderError(message, question);
        scrollToLatest();
      })
      .then(function () {
        setBusy(false);
        el.input.focus();
      });
  }

  /* "New inquiry" clears both halves of the session: the rendered transcript and
   the server-side Conversation. The request is awaited rather than fired and
   forgotten, because a follow-up sent immediately afterwards would otherwise
   resolve against a conversation that has not been dropped yet. */
  function resetChat() {
    var previous = state.sessionId;
    state.transcript = [];
    saveTranscript();
    renderHistory();
    renderEmptyState();
    el.input.value = "";
    el.input.focus();
    if (!previous) {
      return;
    }
    fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: previous })
    }).catch(function () {
      /* Best effort. The next question simply opens a new conversation anyway. */
    });
  }

  /* --- wiring ------------------------------------------------------------ */

  function init() {
    initTheme();

    el.send.addEventListener("click", function () {
      ask(el.input.value);
    });

    el.input.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        ask(el.input.value);
      }
    });

    el.input.addEventListener("input", function () {
      el.send.disabled = state.busy || !el.input.value.trim();
    });

    el.newInquiry.addEventListener("click", resetChat);

    el.themeToggle.addEventListener("click", function () {
      applyTheme(document.body.classList.contains("theme-dark") ? "light" : "dark");
    });

    el.sidebarToggle.addEventListener("click", function () {
      el.sidebar.classList.toggle("is-open");
    });

    el.helpBtn.addEventListener("click", function () {
      el.helpSheet.hidden = false;
    });

    el.helpSheet.addEventListener("click", function (event) {
      if (event.target.hasAttribute("data-close-help")) {
        el.helpSheet.hidden = true;
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        el.helpSheet.hidden = true;
        el.sidebar.classList.remove("is-open");
      }
    });

    fetch("/api/meta")
      .then(function (response) { return response.json(); })
      .then(function (meta) {
        state.meta = meta;
        el.banner.textContent = meta.banner || el.banner.textContent;
        el.footerNote.textContent = meta.footer_note || el.footerNote.textContent;
        if (meta.corpus_note) {
          /* Inside the empty-state template, so it only exists once that has
           * been cloned into the thread. */
          var note = document.getElementById("corpus-note");
          if (note) {
            note.textContent = meta.corpus_note;
          }
        }
        renderSources();
        renderSuggestions();
        renderHelp();
        renderStartupProblems();
      })
      .catch(function () {
        renderSources();
      });

    /* Replay the transcript so a refresh does not lose the visible thread.
     * The server-side conversation is untouched either way. */
    if (state.transcript.length) {
      state.transcript.forEach(function (turn) {
        if (turn.role === "user") {
          renderUser(turn.question, turn.id);
        } else {
          renderAssistant(turn);
        }
      });
    } else {
      renderEmptyState();
    }
    renderHistory();
    el.send.disabled = true;
    el.input.focus();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();