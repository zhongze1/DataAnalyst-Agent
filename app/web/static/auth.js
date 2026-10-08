/* ---------------------------------------------------------------- 认证
 *
 * 三种运行状态（都由 GET /api/auth/me 的响应决定）：
 *   - 200 + user  -> 已登录，正常使用
 *   - 200 + null  -> 认证已关闭（本机模式），直接使用，不显示登录页
 *   - 401         -> 未登录，显示登录页
 */

const LOGIN_CARD_ID = "loginCard";

function showLogin(message) {
  const card = document.getElementById(LOGIN_CARD_ID);
  const app = document.getElementById("appMain");
  if (card) card.hidden = false;
  if (app) app.hidden = true;
  if (message) {
    const hint = document.getElementById("loginError");
    if (hint) hint.textContent = message;
  }
}

function showApp(user) {
  const card = document.getElementById(LOGIN_CARD_ID);
  const app = document.getElementById("appMain");
  if (card) card.hidden = true;
  if (app) app.hidden = false;
  renderUserBar(user);
}

function renderUserBar(user) {
  const bar = document.getElementById("userBar");
  if (!bar) return;
  if (!user) {
    // 本机模式：明确告诉用户「没开认证」，别让人误以为环境安全
    bar.innerHTML =
      `<span class="user-chip warn" title="关闭认证时任何人都能访问本服务">`
      + `本机模式（未启用登录）</span>`;
    return;
  }
  const roleLabel = user.role === "admin" ? "管理员" : "分析师";
  bar.innerHTML =
    `<span class="user-chip">${escapeHtml(user.display_name || user.username)}`
    + `<em>${escapeHtml(roleLabel)}</em></span>`
    + `<button type="button" class="link-btn" id="logoutBtn">退出</button>`;
  document.getElementById("logoutBtn")?.addEventListener("click", doLogout);
}

async function doLogout() {
  if (state.busy) {
    addMessage("assistant", `<p class="err">分析进行中，请先停止再退出登录。</p>`);
    return;
  }
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch (err) {
    // 即使服务端调用失败也要切回登录页：本地 Cookie 可能已失效
  }
  state.sessionId = null;
  localStorage.removeItem("dataanalyst.sessionId");
  showLogin("已退出登录");
}

async function doLogin(event) {
  event.preventDefault();
  const username = document.getElementById("loginUsername")?.value.trim();
  const password = document.getElementById("loginPassword")?.value ?? "";
  const button = document.getElementById("loginBtn");
  const error = document.getElementById("loginError");
  if (button) button.disabled = true;
  if (error) error.textContent = "";
  try {
    const user = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    showApp(user);
    await loadHistory();
    await restoreLastSession();
  } catch (err) {
    if (error) error.textContent = err.message;
  } finally {
    if (button) button.disabled = false;
  }
}

async function checkAuth() {
  let user = null;
  try {
    const response = await fetch("/api/auth/me", { credentials: "same-origin" });
    if (response.status === 401) {
      showLogin();
      return;
    }
    if (response.ok) user = await response.json();
  } catch (err) {
    showLogin("无法连接到服务");
    return;
  }
  if (user) {
    showApp(user);
    await loadHistory();
    await restoreLastSession();
  } else {
    // 认证关闭：本机模式
    showApp(null);
    await loadHistory();
    await restoreLastSession();
  }
}

document.getElementById("loginForm")?.addEventListener("submit", doLogin);
checkAuth();