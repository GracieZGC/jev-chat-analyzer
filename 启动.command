#!/bin/bash
# 恋爱·职场聊天神器 —— 双击启动本地服务并打开页面。
#
# 本脚本要同时兼容 bash 和 zsh（双击 .command 时 Terminal 可能用 zsh 执行），
# 所以只用两者都支持的写法，有两条硬规矩：
#   1) 不要用 read -p —— zsh 里 -p 是协进程，不是提示语；用 printf + read -r。
#   2) 变量后面紧跟中文/全角字符时必须写 ${VAR} —— bash 会把多字节字符的首字节
#      吞进变量名，导致变量变空、后面的字变成乱码。zsh 无此问题，但两边都要过。
#
# 2026-09-23 补的一条：端口被占时不再无条件复用旧服务。以前只要端口有人就
# 直接开页面，结果常常打开的是上一版代码的旧服务 —— 看着像"启动了"，改动
# 却完全没生效。现在会比对版本，不一样就问要不要停掉重起。
cd "$(dirname "$0")" || exit 1

# 手机UI 副本默认跑在 8768，避开原项目的 8767，这样两个版本可以同时开着对照。
# 想改端口：在终端里 `PORT=9000 ./启动.command` 即可。
PORT="${PORT:-8768}"
export PORT
URL="http://127.0.0.1:$PORT"

# 双击打开时 Terminal 给的 PATH 只有 /usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin，
# 不含 Homebrew 目录，会选到系统自带的 Python 3.9。先把它加回来。
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

open_url() {
  if [ -d "/Applications/Google Chrome.app" ]; then
    open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL"
  else
    open "$URL"
  fi
}

listening() {
  lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1
}

# 读两个版本号：第一行 = 当前代码里的 VERSION，第二行 = 端口上真正在跑的版本。
# 探测不到就给 '?'（端口没起、或被别的服务占着都是这个结果）。
versions() {
  "${PY}" - "$PORT" <<'PYEOF'
import json, re, sys, urllib.request
port = sys.argv[1]
try:
    src = open('server.py', encoding='utf-8').read()
    hit = re.search(r"^VERSION\s*=\s*['\"]([^'\"]+)", src, re.M)
    cur = hit.group(1) if hit else '?'
except Exception:
    cur = '?'
run = '?'
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:%s/health' % port, timeout=3) as resp:
        run = json.loads(resp.read().decode('utf-8', 'replace')).get('version', '?')
except Exception:
    pass
print(cur)
print(run)
PYEOF
}

PY=$(command -v python3)
if [ -z "$PY" ]; then
  echo "✗ 没找到 python3。请先安装 Python 3，或手动运行：python3 server.py"
  printf '按回车键关闭窗口… '
  read -r _pause
  exit 1
fi

V=$(versions)
CUR=$(printf '%s\n' "$V" | sed -n 1p)
RUN=$(printf '%s\n' "$V" | sed -n 2p)

if listening; then
  if [ "$RUN" = "$CUR" ]; then
    echo "端口 ${PORT} 上已经有一个 ${CUR} 在跑，直接打开页面。"
    open_url
    exit 0
  fi
  echo "⚠  端口 ${PORT} 被占着，但跑的不是当前代码："
  echo "     正在跑的是「${RUN}」，当前代码是「${CUR}」。"
  echo "     直接打开会看到旧页面，最近的改动不会生效。"
  printf '   要停掉它、重新起一个吗？[回车=停掉重启 / n=就用旧的] '
  read -r _ans
  case "$_ans" in
    n|N)
      echo "   保持不动，打开现有页面。"
      open_url
      exit 0
      ;;
    *)
      OLD_PID=$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t | head -1)
      if [ -n "$OLD_PID" ]; then
        kill "$OLD_PID" 2>/dev/null
        sleep 1
        kill -9 "$OLD_PID" 2>/dev/null
        echo "   已停掉旧进程 ${OLD_PID}。"
      fi
      if listening; then
        echo "✗ 还是停不掉。手动查一下是谁占着：lsof -nP -iTCP:${PORT} -sTCP:LISTEN"
        printf '按回车键关闭窗口… '
        read -r _pause
        exit 1
      fi
      ;;
  esac
fi

echo "恋爱·职场聊天神器 ${CUR} 启动中 → ${URL}"
echo "解释器：${PY}（$("${PY}" --version 2>&1)）"
echo "按 Ctrl+C 停止服务。"
echo ""

"${PY}" server.py &
SERVER_PID=$!

# 等端口真的监听起来再打开页面；最多等 5 秒。
i=0
while [ $i -lt 25 ]; do
  listening && break
  kill -0 "$SERVER_PID" 2>/dev/null || break   # 进程已经挂了，不必再等
  sleep 0.2
  i=$((i + 1))
done

if listening; then
  open_url
else
  echo ""
  echo "✗ 服务没能在 127.0.0.1:${PORT} 上起来，页面不打开。上面的报错就是原因。"
  echo "  排查端口占用：lsof -nP -iTCP:${PORT} -sTCP:LISTEN"
  kill "$SERVER_PID" 2>/dev/null
  printf '按回车键关闭窗口… '
  read -r _pause
  exit 1
fi

wait "$SERVER_PID"
CODE=$?

# Ctrl+C 会让子进程带 130 退出，这是正常停止，不用报警。
if [ "$CODE" -eq 0 ] || [ "$CODE" -eq 130 ]; then
  echo ""
  echo "服务已停止。"
  exit 0
fi

echo ""
echo "服务异常退出（退出码 ${CODE}）。"
printf '按回车键关闭窗口… '
read -r _pause
exit "$CODE"
