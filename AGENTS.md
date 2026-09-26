# 岐问项目协作约定

- 每次修改前端、后端或运行配置后，在交付前执行 `docker compose up --build -d`。
- 随后检查 `docker compose ps` 中前后端健康状态，并确认 `http://127.0.0.1:23000/api/health` 与 `/chat` 可访问。
- 若 Docker 更新或健康检查未完成，明确告知用户，不将代码构建成功等同于服务已更新。
