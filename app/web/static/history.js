/* ---------------------------------------------------------------- 历史会话
 *
 * 会话列表来自 GET /api/sessions（服务端已按最近活跃倒序）。
 * 当前会话 id 存 localStorage，刷新后自动恢复。
 *
 * 依赖 app.js 先加载（共用其 state / dom / api 等全局词法作用域）。
 */

const HISTORY_KEY = "dataanalyst.sessionId";

function relativeTime(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Date.now() - then;
  const min = Math.floor(diff / 60000);
  if (min < 1) return "刚刚";
  if (min < 60) return `${min} 分钟前`;
  const hour = Math.floor(min / 60);
  if (hour < 24) return `${hour} 小时前`;
  const day = Math.floor(hour / 24);
  if (day < 30) return `${day} 天前`;
  return new Date(then).toLocaleDateString("zh-CN");
}

function renderHistory() {
  const list = state.history || [];
  if (!list.length) {
    dom.history.innerHTML = `<p class="empty">还没有历史会话</p>`;
    return;
  }
  dom.history.innerHTML = list
    .map((s) => {
      const active = s.session_id === state.sessionId ? " active" : "";
      const meta = [relativeTime(s.updated_at), s.turn_count ? `${s.turn_count} 轮` : "未提问"].join(" · ");
      return `<div class="hist-item${active}" data-sid="${s.session_id}" title="${escapeHtml(s.filename)}">
        <div class="hist-name">${escapeHtml(s.preview || s.filename)}</div>
        <div class="hist-meta"><span>${escapeHtml(s.filename)}</span><span>${escapeHtml(meta)}</span></div>
      </div>`;
    })
    .join("");
}

async function loadHistory() {
  try {
    state.history = await api("/api/sessions");
    renderHistory();
  } catch (err) {
    dom.history.innerHTML = `<p class="empty">历史加载失败：${escapeHtml(err.message)}</p>`;
  }
}

/* 切换到某个历史会话：拉详情并重渲染整个消息区 */
async function openSession(sessionId) {
  const detail = await api(`/api/sessions/${sessionId}`);
  state.sessionId = sessionId;
  localStorage.setItem(HISTORY_KEY, sessionId);

  dom.fileInfo.className = "file-info";
  dom.fileInfo.innerHTML =
    `<strong>${escapeHtml(detail.filename)}</strong>`
    + `<span class="file-meta">${detail.sheets.length} 个工作表 · ${fmtSize(detail.size_bytes)}`;
  dom.sheetList.innerHTML = detail.sheets
    .map((s) => `<span class="chip">${escapeHtml(s)}</span>`)
    .join("");
  dom.resetBtn.disabled = false;
  setBusy(false);

  dom.messages.innerHTML = "";
  const messages = detail.messages || [];
  for (const message of messages) {
    // 只渲染 user / assistant：tool 消息的结论已包含在 assistant 回复里，
    // 单独显示反而会重复且难懂
    if (message.role === "user") {
      addMessage("user", renderMarkdown(message.content));
    } else if (message.role === "assistant") {
      addMessage("assistant", renderMarkdown(message.content || ""));
    }
  }
  if (!messages.length) {
    addMessage(
      "assistant",
      `<p>已恢复会话 <strong>${escapeHtml(detail.filename)}</strong>，继续提问即可。</p>`
    );
  }
  renderHistory();
}

function bindHistory() {
  dom.history.addEventListener("click", async (e) => {
    const item = e.target.closest(".hist-item");
    if (!item) return;
    if (state.busy) {
      addMessage("assistant", `<p class="err">分析进行中，请先停止再切换会话。</p>`);
      return;
    }
    try {
      await openSession(item.dataset.sid);
    } catch (err) {
      addMessage("assistant", `<p class="err">切换会话失败：${escapeHtml(err.message)}</p>`);
    }
  });
}

/* 刷新页面后恢复上次会话 */
async function restoreLastSession() {
  const saved = localStorage.getItem(HISTORY_KEY);
  if (!saved) return;
  try {
    await openSession(saved);
  } catch {
    // 会话可能已被删除：清掉脏 id，静默忽略
    localStorage.removeItem(HISTORY_KEY);
  }
}

(function initHistory() {
  // 容器缺失必须显式报错：否则 bindHistory() 抛错会静默中断，
  // 表现为面板永远停在「加载中…」，毫无线索
  if (!dom.history) {
    console.error("[history] 页面缺少 #history 容器，历史面板已禁用");
    return;
  }
  try {
    bindHistory();
    loadHistory();
    restoreLastSession();
  } catch (err) {
    console.error("[history] 初始化失败", err);
    dom.history.innerHTML = `<p class="empty">历史面板初始化失败：${escapeHtml(err.message)}</p>`;
  }
})();
