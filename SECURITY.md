# 安全说明

## 已知边界（请务必先读）

### `run_pandas` 沙箱不是安全边界

`run_pandas` 会用 AST 黑名单过滤模型生成的代码，但这只能挡住"直白写法"
（`import`、`open`、`to_csv`、`read_*`、dunder 等）。**已确认存在绕过方式**：

| 绕过手段 | 后果 |
| --- | --- |
| `pd.eval('pd.io.common.os')` / `df.eval(...)` / `df.query(...)` | AST 看不到字符串里的内容，可拿到 `os` 模块 → 执行系统命令 |
| `pd.ExcelFile(path).parse()` / `pd.ExcelWriter(path)` | 未列入黑名单，可读写任意路径文件 |
| `df.apply('to_csv', args=(path,))` | 字符串派发，属性名里不含 `to_csv`，绕过属性链检查 |

**因此：不要把本项目当作多租户隔离手段。** 适用于「提问者是可信的」场景
（本机、内网、单人使用），或部署在容器/虚拟机中并限制其对宿主文件的访问。

彻底修复的方向是把沙箱改成「**子进程 + 容器/seccomp + CPU/内存/时间限额**」，
而不是继续往黑名单里加规则 —— 黑名单永远堵不完。

### 部署建议

- **必须开启认证**（`AGENT_AUTH_ENABLED=true`）并配置强口令的初始管理员；
- 放在 TLS 之后，并置 `AGENT_AUTH_COOKIE_SECURE=true`；
- **只支持单进程部署**：配额与并发的分钟窗口是进程内状态，`--workers N` 会让
  实际额度放大 N 倍；
- `data/` 下有上传的原始 Excel、SQLite 会话库与审计日志，注意备份与访问权限。

## 报告漏洞

如果你发现了安全问题，请**不要**直接开公开 issue。请通过 GitHub 的
[私密漏洞报告](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
功能联系维护者，或在仓库的 Security 标签页提交 Security Advisory。

请包含：受影响的版本、复现步骤、影响评估。我们会在确认后尽快修复并在
CHANGELOG 中致谢。
