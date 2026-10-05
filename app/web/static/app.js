/* 前端逻辑：文件上传 + 对话 + 工具调用轨迹渲染。原生 JS，无构建步骤。 */

const state = {
  sessionId: null,
  sheets: [],
  busy: false,
};

const el = (id) => document.getElementById(id);
const dom = {
  health: el("health"),
  dropzone: el("dropzone"),
  fileInput: el("fileInput"),
  pickBtn: el("pickBtn"),
  progress: el("progress"),
  fileInfo: el("fileInfo"),
  sheetList: el("sheetList"),
  samples: el("samples"),
  messages: el("messages"),
  composer: el("composer"),
  input: el("input"),
  sendBtn: el("sendBtn"),
  resetBtn: el("resetBtn"),
};

/* ---------------------------------------------------------------- 工具函数 */
async function api(path, options = {}) {
  const res = await fetch(path, options);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  // 任何响应都带 X-Request-ID，200 的异常场景（如模型故障）也要能报障
  state.lastRequestId = res.headers.get("X-Request-ID") || null;
  if (!res.ok) {
    // 统一错误体：{error:{code,message,request_id}}；detail 兜底兼容旧格式
    const message = data?.error?.message || data?.detail || `请求失败 (${res.status})`;
    const error = new Error(typeof message === "string" ? message : JSON.stringify(message));
    error.code = data?.error?.code || null;
    error.requestId = data?.error?.request_id || state.lastRequestId;
    throw error;
  }
  return data;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* 错误气泡：带上 request_id，用户报障时直接报这个串就能定位到日志 */
function errorHtml(err) {
  const rid = err.requestId ? `<div class="rid">请求 ID: ${err.requestId}</div>` : "";
  return `<p class="err">⚠ ${escapeHtml(err.message)}</p>${rid}`;
}

function fmtSize(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

/* 极简 Markdown 渲染：先转义再处理，避免 XSS */
function renderMarkdown(text) {
  const escaped = String(text ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const blocks = [];
  let html = escaped.replace(/```([\s\S]*?)```/g, (_, code) => {
    // 答案里的代码默认折叠：用户要的是结论，代码属于「想看才看」的内容。
    // 实际调用的代码在下方「执行过程」面板里，那才是准确的一份。
    blocks.push(
      `<details class="answer-code"><summary>查看代码</summary>`
      + `<pre><code>${code.replace(/^\n/, "")}</code></pre></details>`
    );
    return `\u0000${blocks.length - 1}\u0000`;
  });

  html = html
    .replace(/^### (.+)$/gm, "<h4>$1</h4>")
    .replace(/^## (.+)$/gm, "<h4>$1</h4>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*(?!\s)([^*\n]+?)\*/g, "$1<em>$2</em>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .split(/\n{2,}/)
    .map((block) => {
      // 代码块已被替换成 \u0000N\u0000 占位符：原样保留，交给末尾统一还原。
      // 早前这里返回 null 会把占位符连同代码块一起丢掉，导致答案里的代码块
      // 根本不显示，且没有任何报错。
      if (block.includes("\u0000")) return block;
      const lines = block.split("\n");
      if (lines[0].trim().startsWith("|") && lines.length > 1) {
        const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const head = cells(lines[0]);
        const body = lines.slice(2).filter((l) => l.trim()).map(cells);
        return `<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`
          + `<tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
      }
      if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
        return `<ul>${lines.map((l) => `<li>${l.replace(/^\s*[-*]\s+/, "")}</li>`).join("")}</ul>`;
      }
      if (lines.every((l) => /^\s*\d+\.\s+/.test(l))) {
        return `<ol>${lines.map((l) => `<li>${l.replace(/^\s*\d+\.\s+/, "")}</li>`).join("")}</ol>`;
      }
      const heading = lines[0].match(/^<h4>(.*)<\/h4>$/);
      if (heading) return `<h4>${heading[1]}</h4><p>${lines.slice(1).join(" ")}</p>`;
      return `<p>${lines.join("<br>")}</p>`;
    })
    .filter(Boolean)
    .join("");

  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

/* ---------------------------------------------------------------- 消息渲染 */
function addMessage(role, html, raw = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "我" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;
  wrapper.append(avatar, bubble);
  dom.messages.append(wrapper);
  dom.messages.scrollTop = dom.messages.scrollHeight;
  return bubble;
}

function addError(text) {
  return addMessage("assistant", `<p class="err">⚠ ${text}</p>`);
}

/* 代码类参数：单独渲染成代码块，不与工具名挤在同一行 */
const CODE_ARGS = new Set(["code"]);
/* 短于此行数则默认展开 —— 常见的一行表达式不该再点一次 */
const AUTO_EXPAND_LINES = 4;

function formatArgValue(value) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return `[${value.join(", ")}]`;
  return JSON.stringify(value);
}

function renderStepArgs(args) {
  const entries = Object.entries(args || {});
  const brief = entries
    .filter(([k]) => !CODE_ARGS.has(k))
    .map(([k, v]) => `<span class="arg">${escapeHtml(k)}=${escapeHtml(formatArgValue(v))}</span>`);

  const code = entries.find(([k]) => CODE_ARGS.has(k));
  if (!code) return brief.length ? `<div class="step-args">${brief.join("")}</div>` : "";

  const source = String(code[1] ?? "");
  const lineCount = source.split("\n").length;
  const open = lineCount <= AUTO_EXPAND_LINES ? " open" : "";
  return `<div class="step-args">${brief.join("")}</div>`
    + `<details class="step-code"${open}>`
    + `<summary>计算代码（${lineCount} 行）</summary>`
    + `<pre>${escapeHtml(source)}</pre>`
    + `</details>`;
}

function renderSteps(steps) {
  if (!steps?.length) return "";
  const failed = steps.filter((s) => !s.ok).length;
  const items = steps.map((s) => {
    const status = s.ok
      ? `<span class="step-status ok">成功</span>`
      : `<span class="step-status bad">失败</span>`;
    const body = escapeHtml(s.ok ? s.result : s.error || "");
    return `<div class="step ${s.ok ? "" : "fail"}">
      <div class="step-head">
        <span class="step-no">#${s.step}</span>
        <span class="step-name">${escapeHtml(s.tool)}</span>
        ${status}
      </div>
      ${renderStepArgs(s.arguments)}
      <div class="step-body">${body}</div>
    </div>`;
  }).join("");
  const summary = failed
    ? `执行过程 · ${steps.length} 次工具调用（${failed} 次失败）`
    : `执行过程 · ${steps.length} 次工具调用`;
  return `<details class="steps"><summary>${summary}</summary>${items}</details>`;
}

/* ---------------------------------------------------------------- 上传 */
async function uploadFile(file) {
  if (!file) return;
  dom.progress.classList.remove("hidden");
  try {
    const form = new FormData();
    form.append("file", file);
    const info = await api("/api/files/upload", { method: "POST", body: form });
    state.sessionId = info.file.session_id;
    state.sheets = info.file.sheets;

    dom.fileInfo.className = "file-info";
    dom.fileInfo.innerHTML = `<strong>${info.file.filename}</strong>`
      + `<span class="file-meta">${info.file.sheets.length} 个工作表 · ${fmtSize(info.file.size_bytes)}</span>`;
    dom.sheetList.innerHTML = info.file.sheets.map((s) => `<span class="chip">${s}</span>`).join("");
    dom.resetBtn.disabled = false;
    setBusy(false);
    addMessage("assistant", renderMarkdown(
      `文件 **${info.file.filename}** 上传成功，包含 ${info.file.sheets.length} 个工作表：`
      + `${info.file.sheets.map((s) => `\`${s}\``).join("、")}。\n\n请描述你的分析问题，我会自主读取数据并计算。`
    ));
  } catch (err) {
    setBusy(false);
    addMessage("assistant", errorHtml({ message: `上传失败：${err.message}`, requestId: err.requestId }));
  } finally {
    dom.progress.classList.add("hidden");
    dom.fileInput.value = "";
  }
}

/* ---------------------------------------------------------------- 提问 */
async function send(question) {
  if (state.busy || !question.trim()) return;
  addMessage("user", renderMarkdown(question));
  setBusy(true);
  const thinking = addMessage("assistant", `<span class="thinking">正在分析数据…</span>`);

  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message: question }),
    });
    thinking.innerHTML = renderMarkdown(data.answer) + renderSteps(data.steps);
    if (data.stop_reason === "max_steps") {
      thinking.innerHTML += `<p class="err">已达到最大工具调用次数，可点「清空对话」后重新提问。</p>`;
    } else if (data.stop_reason === "llm_error") {
      // 模型故障：把 request_id 一并给出，方便对照服务端日志排查
      thinking.innerHTML += errorHtml({ message: data.answer, requestId: state.lastRequestId });
    }
  } catch (err) {
    thinking.innerHTML = errorHtml(err);
  } finally {
    setBusy(false);
    dom.input.focus();
  }
}

function setBusy(busy) {
  state.busy = busy;
  dom.sendBtn.disabled = busy || !state.sessionId;
  dom.input.disabled = busy;
  dom.composer.querySelector("button").textContent = busy ? "分析中…" : "发送";
}

/* ---------------------------------------------------------------- 事件绑定 */
dom.pickBtn.addEventListener("click", (e) => { e.preventDefault(); dom.fileInput.click(); });
dom.fileInput.addEventListener("change", (e) => uploadFile(e.target.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dom.dropzone.addEventListener(evt, (e) => { e.preventDefault(); dom.dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((evt) =>
  dom.dropzone.addEventListener(evt, (e) => { e.preventDefault(); dom.dropzone.classList.remove("drag"); })
);
dom.dropzone.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));

dom.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = dom.input.value.trim();
  if (!question) return;
  dom.input.value = "";
  dom.input.style.height = "auto";
  send(question);
});

dom.input.addEventListener("input", () => {
  dom.input.style.height = "auto";
  dom.input.style.height = `${Math.min(dom.input.scrollHeight, 160)}px`;
});

dom.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    dom.composer.requestSubmit();
  }
});

dom.samples.addEventListener("click", (e) => {
  if (e.target.tagName !== "LI" || !state.sessionId) return;
  send(e.target.textContent.trim());
});

dom.resetBtn.addEventListener("click", async () => {
  if (!state.sessionId) return;
  try {
    await api(`/api/chat/reset?session_id=${encodeURIComponent(state.sessionId)}`, { method: "POST" });
    dom.messages.innerHTML = "";
    addMessage("assistant", "<p>对话已清空，数据仍然保留。开始新的分析吧。</p>");
  } catch (err) {
    addMessage("assistant", errorHtml(err));
  }
});

/* ---------------------------------------------------------------- 初始化 */
(async function init() {
  try {
    const health = await api("/api/health");
    dom.health.textContent = `${health.model} · ${health.provider}`;
    if (health.provider === "mock") {
      dom.health.classList.add("mock");
      dom.health.textContent += "（演示模式）";
    }
  } catch {
    dom.health.textContent = "后端未连接";
  }
  setBusy(false);
})();
