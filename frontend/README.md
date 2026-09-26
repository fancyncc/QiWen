# 岐问前端

当前保留淡蓝主题、登录卡片、聊天布局与侧栏。使用 Next.js 16、React 19、Tailwind CSS 4，依赖以 package-lock.json 为准。

后端启动和账号要求见[项目说明](../README.md)。本地在此目录执行 `npm ci`，复制 `.env.local.example` 为 `.env.local`，再运行 `npm run dev`。同源 `/api/*` 通过 `BACKEND_URL` 转发到自有 FastAPI 后端。

`src/lib/backend-contract.ts` 转换会话 id、专家笔记、阶段、画像和历史字段；`api.ts` 处理 HTTP 和错误；`stream.ts` 解析候选协议，只将最终确认的正文展示到页面，并在停止时请求取消运行。登录错误保留服务端提示，受保护接口 401 清理登录态。回退后需重新登录。

样式、登录页、AppShell、ChatWindow、MessageBubble 保持回退前文件内容。当前界面没有模式选择器和摘要管理面板，后端能力不等于已有页面入口。

执行 `npm test`、`npm run build` 验证。Docker 交付需按根目录 AGENTS.md 重建并检查服务健康。字段见[接口约定](../docs/api.md)，范围见[回退记录](../docs/回退记录.md)。
