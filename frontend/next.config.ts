import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // CPU 本地重排可能超过默认 30 秒；保留有限超时，覆盖最多三轮检索。
    proxyTimeout: 900_000,
  },
  // 开发与生产均通过 rewrites 代理后端 FastAPI（:8000），前端同源调用 /api/*，免 CORS
  async rewrites() {
    const backend = process.env.BACKEND_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
