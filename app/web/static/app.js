/* 前端逻辑：文件上传 + 对话 + 工具调用轨迹渲染。原生 JS，无构建步骤。 */

const state = {
  controller: null,
  sessionId: null,
  history: [],
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
  stopBtn: el("stopBtn"),
  history: el("history"),
  exportMd: el("exportMd"),
  exportXlsx: el("exportXlsx"),
  exportLast: el("exportLast"),
  resetBtn: el("resetBtn"),
};

/* ---------------------------------------------------------------- 工具函数 */
/**
 * 统一请求封装。
 *
 * 必须显式声明 Content-Type：body 是 JSON 字符串时浏览器不会自动补，
 * FastAPI 会把请求当成表单数据解析，返回 422「请求参数不合法」。
 * （FormData 上传走浏览器自动生成的 multipart，不在此列。）
 */
async function api(path, options = {}) {
  const opts = { ...options };
  const isFormData = opts.body instanceof FormData;
  if (opts.body != null && !isFormData && !opts.skipJsonHeader) {
    if (!opts.headers) opts.headers = {};
    if (!Object.keys(opts.headers).some((k) => k.toLowerCase() === "content-type")) {
      opts.headers["Content-Type"] = "application/json";
    }
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  // 任何响应都带 X-Request-ID，200 的异常场景（如模型故障）也要能报障
  state.lastRequestId = res.headers.get("X-Request-ID") || null;
  if (res.status === 401 && !path.startsWith("/api/auth/")) {
    // 登录态可能在分析途中过期：直接把人送回登录页，而不是抛一个看不懂的错误
    window.showLogin?.("登录状态已失效，请重新登录");
    const expired = new Error("登录状态已失效，请重新登录");
    expired.code = "AUTH_REQUIRED";
    throw expired;
  }
  if (!res.ok) {
    // 统一错误体：{error:{code,message,request_id}}；detail 兜底兼容旧格式
    let message = data?.error?.message || data?.detail || `请求失败 (${res.status})`;
    // 422 的 detail 是字段级数组，直接丢给用户等于什么都没说。
    // 提取第一条「哪个字段错了」拼进提示，登录这类场景能一眼看出问题。
    if (res.status === 422 && Array.isArray(data?.detail)) {
      const first = data.detail[0];
      if (first) {
        const field = Array.isArray(first.loc) ? first.loc.slice(1).join(".") : "";
        message = field ? `${message}（字段：${field}）` : message;
      }
    }
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
  // requestId 来自响应头 X-Request-ID，而该头允许客户端自带 → 同样不可信，必须转义
  const rid = err.requestId
    ? `<div class="rid">请求 ID: ${escapeHtml(err.requestId)}</div>`
    : "";
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
    // 失败时只显示人话提示；原始错误在服务端日志里
    const body = escapeHtml(s.ok ? s.result : s.error || "工具执行失败");
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
    localStorage.setItem("dataanalyst.sessionId", state.sessionId);

    dom.fileInfo.className = "file-info";
    dom.fileInfo.innerHTML =
      `<strong>${escapeHtml(info.file.filename)}</strong>`
      + `<span class="file-meta">${info.file.sheets.length} 个工作表 · ${fmtSize(info.file.size_bytes)}</span>`;
    // 工作表名来自用户上传的 Excel，属于不可信输入：必须转义后再进 innerHTML，
    // 否则一个把 sheet 名写成 <img src=x onerror=...> 的文件就能在别人浏览器里执行脚本
    dom.sheetList.innerHTML = info.file.sheets
      .map((s) => `<span class="chip">${escapeHtml(s)}</span>`)
      .join("");
    dom.resetBtn.disabled = false;
    setBusy(false);
    loadHistory();
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

/* ---------------------------------------------------------------- 流式 SSE
 *
 * 帧格式：event: <kind>\ndata: <json>\n\n
 * 事件类型：start / tool_args / step / token / done / error
 */
function createSSEParser(onEvent) {
  let buffer = "";
  function handleFrame(frame) {
    if (!frame.trim()) return;
    let kind = "message";
    const dataLines = [];
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith("event:")) kind = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;
    try {
      onEvent(kind, JSON.parse(dataLines.join("\n")));
    } catch (err) {
      console.warn("SSE 帧解析失败", frame, err);
    }
  }
  return {
    push(chunk) {
      buffer += chunk;
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      frames.forEach(handleFrame);
    },
    // 流结束时冲刷缓冲区里可能残留的最后一帧
    flush() {
      if (buffer.trim()) handleFrame(buffer);
      buffer = "";
    },
  };
}

async function streamChat(question, sessionId, { onEvent, signal } = {}) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message: question }),
    signal,
  });

  if (!response.ok || !response.body) {
    // 流开始前就失败（会话不存在等）：解析统一错误体
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
    const error = new Error(
      data?.error?.message || data?.detail || `请求失败 (${response.status})`
    );
    error.requestId = data?.error?.request_id || response.headers.get("X-Request-ID");
    throw error;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = createSSEParser(onEvent);
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    const tail = decoder.decode();
    if (tail) parser.push(tail);
    parser.flush();
  } finally {
    reader.cancel().catch(() => {});
  }
}

/* ---------------------------------------------------------------- 导出报告 */
async function downloadReport(format, lastTurn) {
  if (!state.sessionId) {
    addMessage("assistant", `<p class="err">请先上传文件并进行分析。</p>`);
    return;
  }
  const query = new URLSearchParams({ format });
  if (lastTurn) query.set("last_turn", "true");
  const url = `/api/sessions/${state.sessionId}/export?${query}`;
  const label = format === "md" ? "Markdown" : "Excel";
  try {
    state.lastRequestId = null;
    const response = await fetch(url, { credentials: "same-origin" });
    if (response.status === 401) {
      window.showLogin?.("登录状态已失效，请重新登录");
      return;
    }
    if (!response.ok) {
      let message = `导出失败（${response.status}）`;
      try {
        const data = await response.json();
        message = data?.error?.message || message;
        state.lastRequestId = data?.error?.request_id || null;
      } catch { /* 非 JSON 响应，保留默认文案 */ }
      addMessage("assistant", errorHtml({ message, requestId: state.lastRequestId }));
      return;
    }
    // 从 Content-Disposition 取文件名（服务端已按 RFC 5987 编码中文）
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
    const filename = match ? decodeURIComponent(match[1]) : `report.${format}`;
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    // click() 只是排队一次导航，浏览器实际读取 blob 是异步的。同步 revoke 会让
    // 下载静默失败（无任何报错），且该行为规范未保证、跨浏览器/版本可能翻转。
    // 因此延后释放；anchor 的移除也放进同一批，避免第二个顺序竞争。
    setTimeout(() => {
      anchor.remove();
      URL.revokeObjectURL(objectUrl);
    }, 1000);

    const count = response.headers.get("X-Artifact-Count");
    addMessage("assistant", renderMarkdown(
      `已导出 **${label}** 报告` +
      (count != null ? `，含 ${count} 份数据明细` : "") + "。"
    ));
  } catch (err) {
    addMessage("assistant", errorHtml(err));
  }
}

function bindExport() {
  dom.exportMd?.addEventListener("click", () => downloadReport("md", false));
  dom.exportXlsx?.addEventListener("click", () => downloadReport("xlsx", false));
  dom.exportLast?.addEventListener("click", () => downloadReport("md", true));
}

/* ---------------------------------------------------------------- 提问 */
function renderLiveStep(payload) {
  return `<div class="step live" data-step="${payload.step}">
    <div class="step-head">
      <span class="step-no">#${payload.step}</span>
      <span class="step-name">${escapeHtml(payload.tool)}</span>
      <span class="step-status run">执行中</span>
    </div>
    ${payload.args_summary ? `<div class="step-args"><span class="arg">${escapeHtml(payload.args_summary)}</span></div>` : ""}
  </div>`;
}

function markLiveStep(payload) {
  const row = document.querySelector(`.step.live[data-step="${payload.step}"]`);
  if (!row) return;
  row.classList.remove("live");
  const badge = row.querySelector(".step-status");
  badge.textContent = payload.ok ? "成功" : "失败";
  badge.className = `step-status ${payload.ok ? "ok" : "bad"}`;
  if (payload.duration_ms != null) {
    row.querySelector(".step-args")?.insertAdjacentHTML(
      "beforeend",
      `<span class="arg meta">${payload.duration_ms}ms</span>`
    );
  }
}

async function send(question) {
  if (state.busy || !question.trim()) return;
  addMessage("user", renderMarkdown(question));
  setBusy(true);

  const controller = new AbortController();
  state.controller = controller;

  // 气泡立刻出现，边收事件边往里填 —— 用户全程看得到进展
  const bubble = addMessage("assistant", "");
  const status = document.createElement("div");
  status.className = "stream-status";
  status.textContent = "正在分析…";
  const stepBox = document.createElement("div");
  stepBox.className = "steps live";
  const answerBox = document.createElement("div");
  answerBox.className = "stream-answer";
  bubble.append(status, stepBox, answerBox);

  const scroll = () => { dom.messages.scrollTop = dom.messages.scrollHeight; };
  let stopped = false;

  try {
    await streamChat(question, state.sessionId, {
      signal: controller.signal,
      onEvent(kind, data) {
        switch (kind) {
          case "start":
            status.textContent = "正在分析…";
            break;
          case "tool_args":
            status.textContent = `正在执行 ${data.tool}…`;
            stepBox.insertAdjacentHTML("beforeend", renderLiveStep(data));
            scroll();
            break;
          case "step":
            markLiveStep(data);
            status.textContent = "正在分析…";
            scroll();
            break;
          case "token":
            // 逐字追加。用 textContent 避免把模型输出当 HTML 执行
            answerBox.append(document.createTextNode(data.text));
            scroll();
            break;
          case "done": {
            answerBox.remove();
            stepBox.className = "";
            stepBox.remove();
            bubble.innerHTML =
              renderMarkdown(data.answer) + renderSteps(data.steps || []);
            if (data.stop_reason === "max_steps") {
              bubble.innerHTML += `<p class="err">已达到最大工具调用次数，可点「清空对话」后重新提问。</p>`;
            } else if (data.stop_reason === "llm_error") {
              bubble.innerHTML += errorHtml({
                message: data.answer,
                requestId: state.lastRequestId,
              });
            }
            break;
          }
          case "error":
            bubble.innerHTML = errorHtml({
              message: data.message || "分析失败",
              requestId: state.lastRequestId,
            });
            break;
        }
      },
    });
    if (stopped) {
      bubble.querySelector(".stream-status")?.remove();
      bubble.insertAdjacentHTML(
        "beforeend",
        `<p class="stopped">已停止生成（已收到的内容保留在上方）</p>`
      );
    }
  } catch (err) {
    if (err.name === "AbortError") {
      stopped = true;
      status?.remove();
      bubble.insertAdjacentHTML(
        "beforeend",
        `<p class="stopped">已停止生成（已收到的内容保留在上方）</p>`
      );
    } else {
      status?.remove();
      stepBox?.remove();
      bubble.innerHTML = errorHtml(err);
    }
  } finally {
    state.controller = null;
    setBusy(false);
    dom.input.focus();
  }
}

function stopStream() {
  if (!state.controller) return;
  state.controller.abort();
}

function setBusy(busy) {
  state.busy = busy;
  dom.sendBtn.disabled = busy || !state.sessionId;
  dom.input.disabled = busy;
  dom.sendBtn.textContent = busy ? "分析中…" : "发送";
  dom.stopBtn.hidden = !busy;
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

dom.stopBtn.addEventListener("click", stopStream);
bindExport();

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
